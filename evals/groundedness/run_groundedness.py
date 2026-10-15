"""Groundedness eval: how often does a citation actually support its claim?

This is the number the brief calls for, and it is the number a RAG project
lives or dies on. Four metrics, all reported with 95% confidence intervals
because the sample sizes here are small enough for that to matter:

    citation_rate          factual claims that carry any citation at all
                           (mechanical - no judge involved)
    citation_support_rate  cited claims whose passage actually supports them
                           (judged)
    hallucination_rate     factual claims that are uncited, or cited to a
                           passage that does not support them
    fabricated_citation_rate  answers containing a marker for a passage that
                           was never retrieved (mechanical, and the most
                           damning of the four)

Both the single-shot baseline and the agent are evaluated on the same questions,
because "the agent is better" is a claim that needs a comparison, not an
assertion. The report includes the token and USD cost of each so the quality
difference can be weighed against what it cost.

Run:  python -m evals.groundedness.run_groundedness --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from araa.config import get_settings
from araa.graph import ResearchAgent
from araa.llm import CostLedger, build_llm
from araa.rag import SingleShotRAG
from evals.groundedness.judge import Judgement, LexicalJudge, LLMJudge, Verdict
from evals.harness import (
    EvalArtefact,
    environment_stamp,
    fmt_pct,
    load_question_set,
    mean,
    now_iso,
    require_index,
    wilson_interval,
)


def _rate(successes: int, total: int) -> dict:
    lo, hi = wilson_interval(successes, total)
    return {
        "rate": round(successes / total, 4) if total else 0.0,
        "n": total,
        "successes": successes,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


async def evaluate_mode(
    mode: str,
    questions: list[dict],
    settings,
    llm_judge: LLMJudge,
    lex_judge: LexicalJudge,
) -> dict:
    system = ResearchAgent(settings) if mode == "agent" else SingleShotRAG(settings)
    detail: list[dict] = []

    total_claims = cited_claims = 0
    llm_supported = llm_cited = 0
    lex_supported = lex_judged = 0
    agreements = comparable = 0
    answers_with_fabrication = 0
    judge_errors = unjudgeable = 0

    from araa.retrieval.hybrid import ChunkStore

    chunks = ChunkStore(settings)
    costs: list[float] = []
    tokens: list[int] = []
    latencies: list[float] = []

    try:
        for i, q in enumerate(questions, 1):
            ledger = CostLedger()
            t0 = time.perf_counter()
            try:
                result = await system.answer(q["question"], ledger=ledger)
            except Exception as exc:  # noqa: BLE001
                detail.append({"id": q["id"], "error": f"{type(exc).__name__}: {exc}"})
                print(
                    f"  [{i}/{len(questions)}] {mode}: FAILED {type(exc).__name__}", file=sys.stderr
                )
                continue
            latencies.append(time.perf_counter() - t0)
            costs.append(result.cost.get("cost_usd", 0.0))
            tokens.append(result.cost.get("total_tokens", 0))

            cites = result.citations or {}
            claims = cites.get("claims", [])
            factual = [c for c in claims if c.get("factual")]
            total_claims += len(factual)
            cited = [c for c in factual if c.get("cited")]
            cited_claims += len(cited)
            if cites.get("fabricated_markers"):
                answers_with_fabrication += 1

            # chunk_id -> text, for the judge.
            #
            # The trace is only a hint here. `SingleShotRAG` stores its trace
            # with `include_text=False` (the API default, so responses stay
            # small), which meant every passage handed to the judge was empty,
            # every claim was skipped, and the metric reported 0.0% support
            # rather than "nothing could be judged". Resolving text from the
            # chunk store instead makes the eval independent of how any given
            # mode chose to serialise its trace.
            chunk_text = {
                r["chunk_id"]: r.get("text", "")
                for r in result.retrieval_trace.get("results", [])
                if r.get("text")
            }

            claim_rows = []
            for claim in cited:
                parts = []
                for cid in claim.get("chunk_ids", []):
                    text = chunk_text.get(cid) or ""
                    if not text:
                        chunk = chunks.get(cid)
                        text = chunk.text if chunk else ""
                    if text:
                        parts.append(text)
                passage = "\n\n".join(parts).strip()

                if not passage:
                    # Cannot be judged either way. Counted, never silently
                    # folded into the numerator or the denominator.
                    unjudgeable += 1
                    continue

                llm_j: Judgement = await llm_judge.judge(claim["text"], passage)
                lex_j: Judgement = lex_judge.judge(claim["text"], passage)

                lex_judged += 1
                if lex_j.counts_as_supported:
                    lex_supported += 1

                if llm_j.verdict is Verdict.ERROR:
                    judge_errors += 1
                else:
                    llm_cited += 1
                    if llm_j.counts_as_supported:
                        llm_supported += 1
                    if lex_j.counts_as_supported == llm_j.counts_as_supported:
                        agreements += 1
                    comparable += 1

                claim_rows.append(
                    {
                        "claim": claim["text"][:220],
                        "markers": claim.get("markers"),
                        "llm_verdict": llm_j.verdict.value,
                        "llm_reason": llm_j.reason,
                        "lexical_verdict": lex_j.verdict.value,
                        "lexical_overlap": lex_j.score,
                    }
                )

            detail.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": result.answer[:900],
                    "blocked": result.blocked,
                    "factual_claims": len(factual),
                    "cited_claims": len(cited),
                    "fabricated_markers": cites.get("fabricated_markers", []),
                    "cost_usd": result.cost.get("cost_usd", 0.0),
                    "total_tokens": result.cost.get("total_tokens", 0),
                    "latency_s": round(latencies[-1], 2),
                    "claims": claim_rows,
                }
            )
            print(
                f"  [{i}/{len(questions)}] {mode}: {len(cited)}/{len(factual)} cited, "
                f"{latencies[-1]:.0f}s",
                file=sys.stderr,
            )
    finally:
        await system.aclose()

    ungrounded = (total_claims - cited_claims) + (llm_cited - llm_supported)

    warnings: list[str] = []
    if cited_claims and not llm_cited:
        # The failure that produced a confident 0.0% the first time this ran.
        # A metric with an empty denominator is not a measurement, and it must
        # not be reported as though it were.
        warnings.append(
            f"{mode}: NOT MEASURED - {cited_claims} cited claims but 0 were judged "
            f"({unjudgeable} had no resolvable passage text, {judge_errors} judge errors). "
            "The support and hallucination rates below are meaningless for this mode."
        )
    if unjudgeable:
        warnings.append(
            f"{mode}: {unjudgeable} cited claim(s) had no resolvable passage text and "
            "were excluded from the judged denominators."
        )
    if judge_errors:
        warnings.append(f"{mode}: {judge_errors} judge call(s) returned an unparseable verdict.")

    return {
        "metrics": {
            "citation_rate": _rate(cited_claims, total_claims),
            "citation_support_rate": _rate(llm_supported, llm_cited),
            # Denominator is what was actually judged, not every cited claim.
            # Using cited_claims here made an unmeasured run look like a 0%
            # support rate rather than an absent one.
            "citation_support_rate_lexical": _rate(lex_supported, lex_judged),
            "hallucination_rate": _rate(ungrounded, total_claims),
            "fabricated_citation_rate": _rate(answers_with_fabrication, len(questions)),
            "judge_agreement": _rate(agreements, comparable),
            "judge_errors": judge_errors,
            "unjudgeable_claims": unjudgeable,
            "measured": bool(llm_cited),
            "questions": len(questions),
            "answers_produced": len([d for d in detail if "error" not in d]),
            "mean_cost_usd": round(mean(costs), 6),
            "mean_total_tokens": round(mean([float(t) for t in tokens]), 1),
            "mean_latency_s": round(mean(latencies), 2),
        },
        "warnings": warnings,
        "detail": detail,
    }


async def _amain() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20, help="questions to evaluate")
    ap.add_argument("--modes", nargs="+", default=["single_shot", "agent"])
    ap.add_argument("--set", dest="question_set", default="research_questions")
    ap.add_argument(
        "--judge-backend",
        default=None,
        help=(
            "override the judge's LLM backend (e.g. openai_compat, with ARAA_OPENAI_MODEL "
            "naming a different model) so it differs from the author"
        ),
    )
    args = ap.parse_args()

    s = get_settings()
    require_index(s)

    questions = load_question_set(args.question_set)[: args.limit]
    if not questions:
        raise SystemExit(
            f"Question set {args.question_set!r} is empty. Build it:\n"
            "  python -m evals.questions.build_question_set"
        )

    judge_settings = (
        s if not args.judge_backend else s.model_copy(update={"llm_backend": args.judge_backend})
    )
    judge_llm = build_llm(judge_settings)
    llm_judge = LLMJudge(judge_llm)
    lex_judge = LexicalJudge()

    self_judging = judge_settings.active_model_name == s.active_model_name

    results: dict[str, dict] = {}
    try:
        for mode in args.modes:
            print(f"\n=== {mode} ({len(questions)} questions) ===", file=sys.stderr)
            results[mode] = await evaluate_mode(mode, questions, s, llm_judge, lex_judge)
    finally:
        await judge_llm.aclose()

    notes = [
        "citation_support_rate counts only SUPPORTED. A PARTIAL verdict - the passage "
        "is on topic but the claim overstates it - is counted as unsupported, because a "
        "reader following that citation would be misled.",
        "hallucination_rate = (uncited factual claims + cited-but-unsupported claims) "
        "/ all factual claims.",
        "Non-factual sentences (hedges, transitions, headings, refusals) are excluded "
        "from the denominator; see rag/citations.py:is_factual.",
        "A mode with measured=false was NOT measured: its judged denominator was empty. "
        "Its support and hallucination rates are absent, not zero.",
    ]
    # Per-mode warnings ride in the artefact so they reach docs/results.md, not
    # just the terminal of whoever happened to run it.
    for r in results.values():
        notes.extend(r.get("warnings", []))
    if self_judging:
        notes.append(
            f"WARNING: the judge ({judge_settings.active_model_name}) is the same model "
            "that wrote the answers. A model grading its own output is a sympathetic "
            "judge and these numbers are optimistic. Re-run with --judge-backend "
            "openai_compat pointed at a different model for an independent grade. The "
            "lexical judge is model-independent and is reported alongside as a sanity "
            "floor."
        )

    artefact = EvalArtefact(
        name="groundedness",
        metrics={
            "by_mode": {m: r["metrics"] for m, r in results.items()},
            "judge": {
                "llm_judge": llm_judge.name,
                "lexical_judge": lex_judge.name,
                "self_judging": self_judging,
            },
        },
        detail=[{"mode": m, **d} for m, r in results.items() for d in r["detail"][:20]],
        config={
            "question_set": args.question_set,
            "limit": args.limit,
            **s.generation_signature(),
            **s.retrieval_signature(),
        },
        environment=environment_stamp(s),
        notes=notes,
        started_at=now_iso(),
    )
    path = artefact.write(s)

    print("\n" + "=" * 78)
    print(f"  {'metric':<30}" + "".join(f"{m:>22}" for m in results))
    for key in (
        "citation_rate",
        "citation_support_rate",
        "citation_support_rate_lexical",
        "hallucination_rate",
        "fabricated_citation_rate",
        "judge_agreement",
    ):
        row = f"  {key:<30}"
        for m in results:
            d = results[m]["metrics"][key]
            # Built separately rather than nested inside the f-string: reusing
            # the same quote character inside an f-string expression is a syntax
            # error before Python 3.12, and ruff does not flag it.
            cell = f"{fmt_pct(d['rate'])} (n={d['n']})" if d["n"] else "not measured"
            row += f"{cell:>22}"
        print(row)
    for key, fmt in (
        ("mean_total_tokens", "{:>22.0f}"),
        ("mean_cost_usd", "{:>22.6f}"),
        ("mean_latency_s", "{:>22.1f}"),
    ):
        row = f"  {key:<30}"
        for m in results:
            row += fmt.format(results[m]["metrics"][key])
        print(row)

    for r in results.values():
        for warning in r.get("warnings", []):
            print(f"\n  !! {warning}")

    if self_judging:
        print(
            "\n  NOTE: judge and author are the same model. These support rates are\n"
            "        optimistic. Re-run with --judge-backend openai_compat pointed at\n"
            "        a different model to grade independently."
        )
    print(f"\nartefact: {path}")
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())

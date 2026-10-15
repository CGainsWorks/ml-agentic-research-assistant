"""Token and cost accounting per query.

Answers three questions the brief asks and one it implies:

* What does one query cost, in tokens and in USD?
* Where does that cost go — which graph node is expensive?
* What does the agent cost relative to the single-shot baseline, and is the
  groundedness difference worth the multiple?
* What would the same workload cost on a hosted model? Local inference is free
  at the margin, which makes the USD column 0.00 and hides the real shape of
  the spend. The report therefore prices the *measured token counts* against
  hosted rates as well, clearly labelled as a projection.

Token counts are the provider's own (`prompt_eval_count`/`eval_count` from
Ollama, `usage.prompt_tokens`/`completion_tokens` from an OpenAI-compatible
server). Nothing here is estimated with a character heuristic.

Run:  python -m evals.costs.run_cost_report --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from araa.config import PRICING_USD_PER_MTOK, get_settings
from araa.graph import ResearchAgent
from araa.llm import CostLedger
from araa.rag import SingleShotRAG
from evals.harness import (
    EvalArtefact,
    environment_stamp,
    load_question_set,
    mean,
    now_iso,
    percentile,
    require_index,
)

# Reference price points from `araa.config.PRICING_USD_PER_MTOK`, labelled by
# tier rather than by vendor model ID.
PROJECTION_MODELS = ["hosted-llm-small", "hosted-llm-large"]


def _project(input_tokens: float, output_tokens: float) -> dict[str, float]:
    """Price measured token counts against hosted rates.

    Explicitly a projection: the token counts come from a local model, and a
    different model would tokenise and generate differently. It is here because
    "free" is not a useful planning number for anyone deciding whether to run
    this hosted.
    """
    out = {}
    for model in PROJECTION_MODELS:
        in_p, out_p = PRICING_USD_PER_MTOK[model]
        out[model] = round((input_tokens * in_p + output_tokens * out_p) / 1_000_000, 6)
    return out


async def measure(mode: str, questions: list[dict], settings) -> dict:
    system = ResearchAgent(settings) if mode == "agent" else SingleShotRAG(settings)
    rows: list[dict] = []
    try:
        for i, q in enumerate(questions, 1):
            ledger = CostLedger()
            t0 = time.perf_counter()
            try:
                result = await system.answer(q["question"], ledger=ledger)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [{i}/{len(questions)}] {mode}: FAILED {type(exc).__name__}", file=sys.stderr
                )
                continue
            wall = time.perf_counter() - t0
            cost = result.cost
            rows.append(
                {
                    "id": q["id"],
                    "llm_calls": cost.get("llm_calls", 0),
                    "input_tokens": cost.get("input_tokens", 0),
                    "output_tokens": cost.get("output_tokens", 0),
                    "total_tokens": cost.get("total_tokens", 0),
                    "cost_usd": cost.get("cost_usd", 0.0),
                    "wall_s": round(wall, 2),
                    "llm_latency_s": cost.get("llm_latency_s", 0.0),
                    "by_node": cost.get("by_node", {}),
                    "retrieval_ms": result.timings_ms.get("retrieval")
                    or result.timings_ms.get("retrieve"),
                }
            )
            print(
                f"  [{i}/{len(questions)}] {mode}: {cost.get('total_tokens', 0)} tok, "
                f"{cost.get('llm_calls', 0)} calls, {wall:.0f}s",
                file=sys.stderr,
            )
    finally:
        await system.aclose()

    if not rows:
        return {"metrics": {"queries": 0}, "detail": []}

    node_totals: dict[str, dict] = {}
    for r in rows:
        for node, v in r["by_node"].items():
            b = node_totals.setdefault(node, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            b["calls"] += v["calls"]
            b["input_tokens"] += v["input_tokens"]
            b["output_tokens"] += v["output_tokens"]

    n = len(rows)
    mean_in = mean([float(r["input_tokens"]) for r in rows])
    mean_out = mean([float(r["output_tokens"]) for r in rows])

    return {
        "metrics": {
            "queries": n,
            "mean_llm_calls": round(mean([float(r["llm_calls"]) for r in rows]), 2),
            "mean_input_tokens": round(mean_in, 1),
            "mean_output_tokens": round(mean_out, 1),
            "mean_total_tokens": round(mean([float(r["total_tokens"]) for r in rows]), 1),
            "p95_total_tokens": round(
                percentile([float(r["total_tokens"]) for r in rows], 0.95), 1
            ),
            "mean_cost_usd_actual": round(mean([r["cost_usd"] for r in rows]), 6),
            "mean_wall_s": round(mean([r["wall_s"] for r in rows]), 2),
            "p95_wall_s": round(percentile([r["wall_s"] for r in rows], 0.95), 2),
            "projected_cost_usd_per_query": _project(mean_in, mean_out),
            "projected_cost_usd_per_1000_queries": {
                k: round(v * 1000, 2) for k, v in _project(mean_in, mean_out).items()
            },
            "tokens_by_node": {
                node: {
                    **v,
                    "mean_tokens_per_query": round((v["input_tokens"] + v["output_tokens"]) / n, 1),
                    "share_of_tokens": round(
                        (v["input_tokens"] + v["output_tokens"])
                        / max(1, sum(r["total_tokens"] for r in rows)),
                        4,
                    ),
                }
                for node, v in sorted(node_totals.items())
            },
        },
        "detail": rows,
    }


async def _amain() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["single_shot", "agent"])
    ap.add_argument("--set", dest="question_set", default="research_questions")
    args = ap.parse_args()

    s = get_settings()
    require_index(s)
    questions = load_question_set(args.question_set)[: args.limit]
    if not questions:
        raise SystemExit("No questions. Run: python -m evals.questions.build_question_set")

    results = {}
    for mode in args.modes:
        print(f"\n=== {mode} ===", file=sys.stderr)
        results[mode] = await measure(mode, questions, s)

    ratio = None
    if all(results[m]["metrics"].get("queries") for m in ("single_shot", "agent") if m in results):
        base = results["single_shot"]["metrics"]["mean_total_tokens"]
        agent = results["agent"]["metrics"]["mean_total_tokens"]
        ratio = round(agent / base, 2) if base else None

    artefact = EvalArtefact(
        name="costs",
        metrics={
            "by_mode": {m: r["metrics"] for m, r in results.items()},
            "agent_token_multiple_vs_single_shot": ratio,
            "pricing_used_for_projection": {m: PRICING_USD_PER_MTOK[m] for m in PROJECTION_MODELS},
        },
        detail=[{"mode": m, **d} for m, r in results.items() for d in r["detail"]],
        config={"limit": args.limit, **s.generation_signature()},
        environment=environment_stamp(s),
        notes=[
            "Token counts are reported by the provider, not estimated from characters.",
            "mean_cost_usd_actual is 0.00 on a local model by construction: local "
            "inference has no marginal API cost. The electricity is real and is not "
            "measured here.",
            "projected_cost_usd_* prices the *measured* token counts at hosted rates. "
            "It is a planning figure, not a measurement: a different model would "
            "tokenise and generate differently.",
            "Projection price points are labelled by tier (hosted-llm-small, "
            "hosted-llm-large) and shown with their USD per million token rates. Model "
            "identifiers are withheld.",
            "Wall time is dominated by hardware. On the dev box used here the GPU was "
            "occupied, so Ollama ran on CPU. Treat latency as a shape, not a benchmark.",
        ],
        started_at=now_iso(),
    )
    path = artefact.write(s)

    print("\n" + "=" * 78)
    print(f"  {'':<34}" + "".join(f"{m:>20}" for m in results))
    for key, fmt in (
        ("mean_llm_calls", "{:>20.2f}"),
        ("mean_input_tokens", "{:>20.0f}"),
        ("mean_output_tokens", "{:>20.0f}"),
        ("mean_total_tokens", "{:>20.0f}"),
        ("p95_total_tokens", "{:>20.0f}"),
        ("mean_wall_s", "{:>20.1f}"),
    ):
        row = f"  {key:<34}"
        for m in results:
            row += fmt.format(results[m]["metrics"].get(key, 0))
        print(row)

    for model in PROJECTION_MODELS:
        row = f"  {'proj. $/1k queries (' + model + ')':<34}"
        for m in results:
            row += "{:>20.2f}".format(
                results[m]["metrics"].get("projected_cost_usd_per_1000_queries", {}).get(model, 0)
            )
        print(row)

    if ratio:
        print(f"\n  agent costs {ratio}x the tokens of the single-shot baseline")
        print("  weigh that against the groundedness delta in eval-groundedness.json")

    for mode, r in results.items():
        nodes = r["metrics"].get("tokens_by_node", {})
        if nodes:
            print(f"\n  {mode} token share by node:")
            for node, v in sorted(nodes.items(), key=lambda kv: -kv[1]["share_of_tokens"]):
                print(
                    f"    {node:<12} {v['share_of_tokens']:>6.1%}  ({v['mean_tokens_per_query']:.0f} tok/query)"
                )

    print(f"\nartefact: {path}")
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())

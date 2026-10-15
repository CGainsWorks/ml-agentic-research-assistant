"""Render docs/results.md from the eval artefacts.

This is the mechanism behind "every number quoted in the README is reproducible
from a script in this repo". Numbers are never typed into the README by hand:
an eval writes a JSON artefact, this renders those artefacts to markdown, and
the README links to the result. If an eval has not been run, the corresponding
section says so rather than showing a stale figure.

Run:  python -m evals.report
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from araa.config import get_settings
from evals.harness import EvalArtefact, fmt_pct

DOCS = Path(__file__).resolve().parents[1] / "docs"

MISSING = "_Not run yet. Produce it with `{cmd}`._"


def _ci(d: dict) -> str:
    if not isinstance(d, dict) or "ci95" not in d:
        return ""
    lo, hi = d["ci95"]
    return f" (95% CI {fmt_pct(lo)}–{fmt_pct(hi)}, n={d['n']})"


def _rate_cell(d: dict) -> str:
    if not isinstance(d, dict):
        return "—"
    # An empty denominator is not a rate of zero. Rendering it as "0.0%" is how
    # an unmeasured run gets read as a measured failure — which happened here
    # before the eval learned to distinguish the two.
    if not d.get("n"):
        return "**not measured**"
    return f"{fmt_pct(d.get('rate', 0))}{_ci(d)}"


def section_retrieval(a: dict | None) -> list[str]:
    out = ["## Retrieval", ""]
    if not a:
        return out + [MISSING.format(cmd="make eval-retrieval"), ""]

    for set_name, res in a["metrics"].items():
        lift = res["lift"]
        out += [
            f"### Question set: `{set_name}` (n={res['questions']})",
            "",
            f"Headline metric: **{lift['metric']}**.",
            "",
            "| Configuration | chunk hit@5 | doc hit@3 | doc MRR | chunk nDCG@10 | latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, cfg in res["configs"].items():
            m = cfg["metrics"]
            out.append(
                f"| `{name}` | {fmt_pct(m.get('chunk_hit@5', 0))} | {fmt_pct(m.get('doc_hit@3', 0))} "
                f"| {m.get('doc_mrr', 0):.3f} | {m.get('chunk_ndcg@10', 0):.3f} "
                f"| {m.get('latency_ms_mean', 0):.0f} |"
            )
        out += [
            "",
            "**Lift**",
            "",
            "| Comparison | Before | After | Absolute | Relative | Significant? |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for label, key in (
            ("Hybrid vs vector-only", "hybrid_vs_vector_only"),
            ("Reranker vs vector-only", "rerank_vs_vector_only"),
            ("Reranker vs hybrid", "rerank_vs_hybrid"),
        ):
            d = lift[key]
            rel = "—" if d["relative"] is None else f"{d['relative']:+.1%}"
            sig = "—"
            if "intervals_overlap" in d:
                sig = "no — 95% CIs overlap" if d["intervals_overlap"] else "yes"
            out.append(
                f"| {label} | {d['before']:.3f} | {d['after']:.3f} | {d['absolute']:+.4f} | {rel} | {sig} |"
            )
        out += [
            "",
            f"Reranking costs **+{lift['rerank_latency_cost_ms']:.0f} ms/query** on CPU.",
            "",
        ]

    if a.get("notes"):
        out += ["**Notes**", ""] + [f"- {n}" for n in a["notes"]] + [""]
    return out


def section_chunking(a: dict | None) -> list[str]:
    out = ["## Chunking strategy comparison", ""]
    if not a:
        return out + [MISSING.format(cmd="make eval-chunking"), ""]

    m_all = a["metrics"]
    out += [
        "Same corpus, same queries, one isolated index per strategy.",
        "",
        "Primary metric is **doc hit@3** — the gold set is document-level, so chunk-level",
        "nDCG is confounded by how many chunks a strategy emits.",
        "",
        "| Strategy | Chunks | Mean chars | **doc hit@3** | chunk hit@5 | chunk nDCG@10 | Context tokens (approx) | Ingest (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in m_all["by_strategy"].items():
        m, ix = r["retrieval"], r["index"]
        out.append(
            f"| `{name}` | {ix['chunks']} | {ix['chunk_chars_mean']:.0f} "
            f"| **{fmt_pct(m.get('doc_hit@3', 0))}** "
            f"| {fmt_pct(m['chunk_hit@5'])} | {m['chunk_ndcg@10']:.3f} "
            f"| {m['context_tokens_approx']:.0f} | {ix['ingest_seconds']:.0f} |"
        )

    # Derived here rather than read from the artefact so the report also works
    # on artefacts written before these summary fields existed.
    by_strategy = m_all["by_strategy"]
    doc_scores = {k: v["retrieval"].get("doc_hit@3", 0.0) for k, v in by_strategy.items()}
    best_doc = max(doc_scores, key=doc_scores.get)
    tied = sorted(k for k, v in doc_scores.items() if v == doc_scores[best_doc])
    separable = len(tied) < len(by_strategy)
    cheapest = min(by_strategy, key=lambda k: by_strategy[k]["retrieval"]["context_tokens_approx"])
    fewest = min(by_strategy, key=lambda k: by_strategy[k]["index"]["chunks"])
    m_all = {
        **m_all,
        "best_doc_hit@3": best_doc,
        "tied_on_doc_hit@3": tied,
        "separable_on_quality": separable,
        "lowest_context_cost": cheapest,
        "fewest_chunks": fewest,
    }

    out.append("")
    if tied and not separable:
        out += [
            "**All strategies tie on the primary metric.** This question set cannot "
            "separate them on retrieval quality, so the choice is a cost decision: "
            f"lowest context cost is **`{m_all['lowest_context_cost']}`**, fewest chunks "
            f"is **`{m_all.get('fewest_chunks', '—')}`**.",
            "",
            "Reporting `best nDCG@10` here would have declared a winner on an artefact "
            "of chunk count rather than on retrieval quality.",
        ]
    else:
        out += [
            f"Best doc hit@3: **`{m_all.get('best_doc_hit@3', '—')}`**. "
            f"Lowest context cost: **`{m_all['lowest_context_cost']}`**.",
        ]
    out.append("")
    if a.get("notes"):
        out += [f"- {n}" for n in a["notes"]] + [""]
    return out


def section_groundedness(a: dict | None) -> list[str]:
    out = ["## Citation groundedness", ""]
    if not a:
        return out + [MISSING.format(cmd="make eval-groundedness"), ""]

    by_mode = a["metrics"]["by_mode"]
    modes = list(by_mode)
    out += [
        "Does a citation actually support the claim it is attached to?",
        "",
        "| Metric | " + " | ".join(f"`{m}`" for m in modes) + " |",
        "| --- | " + " | ".join("---:" for _ in modes) + " |",
    ]
    for label, key in (
        ("Citation rate (claims carrying a citation)", "citation_rate"),
        ("Citation support rate (LLM judge)", "citation_support_rate"),
        ("Citation support rate (lexical floor)", "citation_support_rate_lexical"),
        ("Hallucination rate (lower is better)", "hallucination_rate"),
        ("Answers with a fabricated marker", "fabricated_citation_rate"),
        ("Judge agreement (LLM vs lexical)", "judge_agreement"),
    ):
        out.append(
            f"| {label} | " + " | ".join(_rate_cell(by_mode[m].get(key)) for m in modes) + " |"
        )
    for label, key, fmt in (
        ("Mean tokens per query", "mean_total_tokens", "{:.0f}"),
        ("Mean latency (s)", "mean_latency_s", "{:.1f}"),
    ):
        out.append(
            f"| {label} | " + " | ".join(fmt.format(by_mode[m].get(key, 0)) for m in modes) + " |"
        )

    judge = a["metrics"].get("judge", {})
    out += [
        "",
        f"Judge: `{judge.get('llm_judge')}` plus `{judge.get('lexical_judge')}` as a floor.",
        "",
    ]
    if judge.get("self_judging"):
        out += [
            "> **The judge is the same model that wrote the answers.** A model grading its "
            "own output is a sympathetic judge, so treat these support rates as optimistic. "
            "Re-run with `--judge-backend openai_compat` pointed at a different model for an "
            "independent grade. The lexical floor is model-independent and is reported "
            "alongside for exactly this reason.",
            "",
        ]
    if a.get("notes"):
        out += [f"- {n}" for n in a["notes"]] + [""]
    return out


def section_adversarial(a: dict | None) -> list[str]:
    out = ["## Adversarial prompt injection", ""]
    if not a:
        return out + [MISSING.format(cmd="make eval-adversarial"), ""]

    m = a["metrics"]
    if det := m.get("detector"):
        out += [
            "### Tier 1 — detector (deterministic, no model, runs in CI)",
            "",
            "| Metric | Result |",
            "| --- | ---: |",
            f"| Attack detection rate | {_rate_cell(det['detection_rate'])} |",
            f"| Block rate on must-block attacks | {_rate_cell(det['block_rate_on_must_block'])} |",
            f"| **False positives on benign controls** | {_rate_cell(det['false_positive_rate'])} |",
            f"| Obfuscated attacks caught after normalisation | {det.get('obfuscated_attacks_caught', 0)} |",
            "",
            "The false-positive row is the one that matters most. This corpus contains "
            "computer-science papers, some of which are *about* prompt injection. A detector "
            "that blocks those scores well on the attack set and makes the corpus unusable.",
            "",
        ]
    if e2e := m.get("end_to_end"):
        out += [
            "### Tier 2 — end to end (payload spliced into a real retrieved passage)",
            "",
            "| Metric | Result |",
            "| --- | ---: |",
            f"| Attacks run | {e2e['attacks_run']} |",
            f"| **Attack success rate** (model complied — lower is better) | {_rate_cell(e2e['attack_success_rate'])} |",
            f"| Blocked outright | {_rate_cell(e2e['blocked_rate'])} |",
            f"| Resisted without blocking (answered anyway) | {e2e['resisted_without_blocking']} |",
            f"| System-prompt canary leaks | {e2e['canary_leaks']} |",
            "",
            "Resisting without blocking is the *best* outcome: the user still gets an answer "
            "and the poisoned passage is reported. Any canary leak is a build failure.",
            "",
        ]
    if a.get("notes"):
        out += [f"- {n}" for n in a["notes"]] + [""]
    return out


def section_costs(a: dict | None) -> list[str]:
    out = ["## Cost per query", ""]
    if not a:
        return out + [MISSING.format(cmd="make eval-cost"), ""]

    by_mode = a["metrics"]["by_mode"]
    modes = list(by_mode)
    out += [
        "| Metric | " + " | ".join(f"`{m}`" for m in modes) + " |",
        "| --- | " + " | ".join("---:" for _ in modes) + " |",
    ]
    for label, key, fmt in (
        ("LLM calls per query", "mean_llm_calls", "{:.2f}"),
        ("Input tokens", "mean_input_tokens", "{:.0f}"),
        ("Output tokens", "mean_output_tokens", "{:.0f}"),
        ("Total tokens", "mean_total_tokens", "{:.0f}"),
        ("Total tokens (p95)", "p95_total_tokens", "{:.0f}"),
        ("Wall time (s)", "mean_wall_s", "{:.1f}"),
        ("Actual USD (local model)", "mean_cost_usd_actual", "{:.4f}"),
    ):
        out.append(
            f"| {label} | " + " | ".join(fmt.format(by_mode[m].get(key, 0)) for m in modes) + " |"
        )

    for model, (in_p, out_p) in a["metrics"].get("pricing_used_for_projection", {}).items():
        out.append(
            f"| Projected USD per 1,000 queries on `{model}` "
            f"(${in_p:.2f} in, ${out_p:.2f} out per Mtok) | "
            + " | ".join(
                "{:.2f}".format(
                    by_mode[m].get("projected_cost_usd_per_1000_queries", {}).get(model, 0)
                )
                for m in modes
            )
            + " |"
        )

    if ratio := a["metrics"].get("agent_token_multiple_vs_single_shot"):
        out += ["", f"The agent uses **{ratio}x** the tokens of the single-shot baseline.", ""]

    for mode in modes:
        nodes = by_mode[mode].get("tokens_by_node", {})
        if not nodes:
            continue
        out += [
            f"**Where `{mode}` spends its tokens**",
            "",
            "| Node | Share | Tokens/query |",
            "| --- | ---: | ---: |",
        ]
        for node, v in sorted(nodes.items(), key=lambda kv: -kv[1]["share_of_tokens"]):
            out.append(
                f"| `{node}` | {v['share_of_tokens']:.1%} | {v['mean_tokens_per_query']:.0f} |"
            )
        out.append("")

    if a.get("notes"):
        out += [f"- {n}" for n in a["notes"]] + [""]
    return out


def main() -> int:
    s = get_settings()
    artefacts = {
        name: EvalArtefact.load(name, s)
        for name in ("retrieval", "chunking", "groundedness", "adversarial", "costs")
    }

    env = next((a["environment"] for a in artefacts.values() if a), {})
    corpus = env.get("corpus", {})
    index = env.get("index", {})

    lines = [
        "# Results",
        "",
        "> Generated by `python -m evals.report` from the JSON artefacts in `data/runs/`.",
        "> Do not edit by hand — regenerate it. Every number below is produced by a",
        "> script in this repository; the command that produces each section is named",
        "> in that section.",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "## Corpus and index under test",
        "",
        "| | |",
        "| --- | --- |",
        f"| Documents | {corpus.get('documents', '—')} |",
        f"| Licences | {json.dumps(corpus.get('licence_counts', {}))} |",
        f"| Chunks indexed | {index.get('chunks', '—')} |",
        f"| Chunking | `{(index.get('config') or {}).get('chunk_strategy', '—')}` |",
        f"| Embedding model | `{(index.get('config') or {}).get('embedding_model', '—')}` |",
        f"| Reranker | `{(index.get('config') or {}).get('reranker_model', '—')}` |",
        "",
        "Every document's licence is recorded in `data/corpus/LICENCES.md`.",
        "",
        "---",
        "",
    ]

    lines += section_retrieval(artefacts["retrieval"])
    lines += ["---", ""]
    lines += section_chunking(artefacts["chunking"])
    lines += ["---", ""]
    lines += section_groundedness(artefacts["groundedness"])
    lines += ["---", ""]
    lines += section_adversarial(artefacts["adversarial"])
    lines += ["---", ""]
    lines += section_costs(artefacts["costs"])

    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "results.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    have = [k for k, v in artefacts.items() if v]
    missing = [k for k, v in artefacts.items() if not v]
    print(f"wrote {path}")
    print(f"  sections with data: {', '.join(have) if have else 'none'}")
    if missing:
        print(f"  not yet run: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

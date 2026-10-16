"""Every published number traces to the committed docs/results.json.

`data/runs/` is gitignored, so the raw eval artefacts are not in the
repository. `docs/results.json` is: `python -m evals.report --from-runs`
condenses the artefacts into it, and `docs/results.md` is rendered from it.
These tests pin the chain at both ends:

* `docs/results.md` must be exactly what `docs/results.json` renders to;
* the figures the README quotes must be the ones in `docs/results.json`;
* the corpus size must be one number everywhere it is configured.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from araa.config import PRICING_USD_PER_MTOK, Settings
from evals.report import REPORT_PATH, SUMMARY_PATH, chunking_verdict, load_summary, render

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


def _summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip("docs/ not present in this build context")
    return load_summary()


def _readme() -> str:
    if not README.exists():
        pytest.skip("README.md not present in this build context")
    return README.read_text(encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def test_results_md_is_rendered_from_results_json():
    assert REPORT_PATH.read_text(encoding="utf-8") == render(
        _summary()
    ), "docs/results.md is stale. Regenerate it: python -m evals.report"


def test_every_section_has_data_and_provenance():
    sections = _summary()["sections"]
    for name, section in sections.items():
        assert section, f"{name} has no data in docs/results.json"
        prov = section["provenance"]
        assert prov["artefact"] == f"data/runs/eval-{name}.json"
        assert re.fullmatch(r"[0-9a-f]{64}", prov["sha256"])
        assert prov["finished_at"]


def test_results_json_holds_no_local_paths():
    text = SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.exists() else ""
    for needle in (":\\\\", "/Users/", "/home/", "/data/"):
        assert needle not in text, f"docs/results.json contains a local path ({needle!r})"


def test_cost_projections_are_labelled_by_tier():
    pricing = _summary()["sections"]["costs"]["metrics"]["pricing_used_for_projection"]
    assert set(pricing) <= set(PRICING_USD_PER_MTOK)
    for label, rates in pricing.items():
        assert tuple(rates) == PRICING_USD_PER_MTOK[label]


def readme_figures(s: dict) -> list[str]:
    """Strings the README must contain, each built from docs/results.json."""
    sec = s["sections"]
    g = sec["groundedness"]["metrics"]["by_mode"]
    ss, ag = g["single_shot"], g["agent"]
    ret = sec["retrieval"]["metrics"]["retrieval_queries"]
    cfg = {k: v["metrics"] for k, v in ret["configs"].items()}
    lift = ret["lift"]
    det = sec["adversarial"]["metrics"]["detector"]
    cost = sec["costs"]["metrics"]
    cm = cost["by_mode"]
    nodes = cm["agent"]["tokens_by_node"]
    chunking = sec["chunking"]["metrics"]["by_strategy"]
    small = cost["pricing_used_for_projection"]["hosted-llm-small"]
    multiple = cost["agent_token_multiple_vs_single_shot"]
    critique_revise = nodes["critique"]["share_of_tokens"] + nodes["revise"]["share_of_tokens"]

    def rate(d: dict) -> str:
        return pct(d["rate"])

    def frac(d: dict) -> str:
        return f"**{100 * d['rate']:.0f}%** ({d['successes']}/{d['n']})"

    out = [
        f"| Citation support rate (LLM judge) | **{rate(ss['citation_support_rate'])}** | **{rate(ag['citation_support_rate'])}** |",
        f"| Citation support rate (lexical floor, model-independent) | **{rate(ss['citation_support_rate_lexical'])}** | **{rate(ag['citation_support_rate_lexical'])}** |",
        f"| Hallucination rate (lower is better) | **{rate(ss['hallucination_rate'])}** | **{rate(ag['hallucination_rate'])}** |",
        f"| Tokens per query | {ss['mean_total_tokens']:,.0f} | {ag['mean_total_tokens']:,.0f} |",
        f"at {multiple:.1f}x the tokens",
        f"{ss['citation_support_rate']['n']} and {ag['citation_support_rate']['n']} judged claims",
        f"{100 * critique_revise:.1f}% of the agent's tokens",
        f"{s['corpus']['documents']}-paper corpus, {s['index']['chunks']:,} chunks, "
        f"{ret['questions']} deterministic queries",
        f"| vector only | {pct(cfg['vector_only']['doc_hit@3'])} | {pct(cfg['vector_only']['chunk_hit@5'])} | {cfg['vector_only']['latency_ms_mean']:.0f} ms |",
        f"| BM25 only | {pct(cfg['bm25_only']['doc_hit@3'])} | {pct(cfg['bm25_only']['chunk_hit@5'])} | {cfg['bm25_only']['latency_ms_mean']:.0f} ms |",
        f"| hybrid (RRF) | {pct(cfg['hybrid']['doc_hit@3'])} | {pct(cfg['hybrid']['chunk_hit@5'])} | {cfg['hybrid']['latency_ms_mean']:.0f} ms |",
        f"| **hybrid + reranker** | **{pct(cfg['hybrid_reranked']['doc_hit@3'])}** | **{pct(cfg['hybrid_reranked']['chunk_hit@5'])}** | {cfg['hybrid_reranked']['latency_ms_mean']:.0f} ms |",
        f"{lift['rerank_vs_vector_only']['relative']:+.1%} relative on doc hit@3",
        f"+{lift['rerank_latency_cost_ms'] / 1000:.1f} s/query",
        f"chunk nDCG@10 of {cfg['hybrid_reranked']['chunk_ndcg@10']:.3f} against "
        f"{cfg['vector_only']['chunk_ndcg@10']:.3f} for vector only",
        f"doc hit@3 went from {pct(cfg['vector_only']['doc_hit@3'])} to "
        f"{pct(cfg['hybrid_reranked']['doc_hit@3'])}",
        f"| Attack detection rate | {frac(det['detection_rate'])} |",
        f"| Block rate on must-block attacks | {frac(det['block_rate_on_must_block'])} |",
        f"| **False positives on benign controls** | {frac(det['false_positive_rate'])} |",
        f"| LLM calls | {cm['single_shot']['mean_llm_calls']:.2f} | {cm['agent']['mean_llm_calls']:.2f} |",
        f"| Total tokens | {cm['single_shot']['mean_total_tokens']:,.0f} | {cm['agent']['mean_total_tokens']:,.0f} |",
        f"| Wall time | {cm['single_shot']['mean_wall_s']:.1f} s | {cm['agent']['mean_wall_s']:.1f} s |",
        f"(${small[0]:g} in, ${small[1]:g} out per Mtok) | "
        f"${cm['single_shot']['projected_cost_usd_per_1000_queries']['hosted-llm-small']:.2f} | "
        f"${cm['agent']['projected_cost_usd_per_1000_queries']['hosted-llm-small']:.2f} |",
        f"(critique {pct(nodes['critique']['share_of_tokens'])}, revise "
        f"{pct(nodes['revise']['share_of_tokens'])}, draft only {pct(nodes['draft']['share_of_tokens'])})",
    ]
    v = chunking_verdict(chunking)
    for name, r in chunking.items():
        label = f"`{name}` (default)" if name == "recursive" else f"`{name}`"
        chunks = r["index"]["chunks"]
        chunks_cell = f"**{chunks}**" if name == v["fewest_chunks"] else str(chunks)
        tokens = f"{r['retrieval']['context_tokens_approx']:.0f}"
        tokens_cell = f"**{tokens}**" if name == v["lowest_context_cost"] else tokens
        out.append(
            f"| {label} | {chunks_cell} | **{pct(r['retrieval']['doc_hit@3'])}** "
            f"| {r['retrieval']['chunk_ndcg@10']:.3f} | {tokens_cell} |"
        )
    return out


def test_readme_quotes_the_committed_figures():
    readme = _readme()
    missing = [f for f in readme_figures(_summary()) if f not in readme]
    assert not missing, "README figures not found (or out of date):\n" + "\n".join(missing)


def _fetch_corpus_default() -> int:
    script = REPO_ROOT / "scripts" / "fetch_corpus.py"
    if not script.exists():
        pytest.skip("scripts/ not present in this build context")
    match = re.search(r"^DEFAULT_LIMIT = (\d+)$", script.read_text(encoding="utf-8"), re.M)
    assert match, "scripts/fetch_corpus.py must define DEFAULT_LIMIT"
    return int(match.group(1))


def test_corpus_size_is_one_number():
    """The published results, the package default, the fetch script and the
    compose bootstrap must all mean the same corpus."""
    default = Settings.model_fields["corpus_target_docs"].default
    assert _fetch_corpus_default() == default
    corpus = _summary()["corpus"]
    assert corpus["documents"] == default
    assert corpus["fetch_params"]["limit"] == default

    bootstrap = (REPO_ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert (
        "corpus_target_docs" in bootstrap
    ), "bootstrap.sh must read the corpus size from araa.config"
    assert not re.search(r"DOCS=\"\$\{[A-Z_]+:-\d+\}\"", bootstrap), "bootstrap.sh hardcodes a size"

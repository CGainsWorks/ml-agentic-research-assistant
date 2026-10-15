"""Retrieval eval: does the pipeline find the right chunk, and does the
reranker earn its latency?

Four configurations, identical index, identical queries:

    vector_only        bge-small bi-encoder, cosine
    bm25_only          BM25 keyword
    hybrid             both, fused with RRF, no reranking
    hybrid_reranked    the above, then a bge-reranker cross-encoder

The headline number the brief asks for — "a measured lift from the reranker
over vector-only" — is `lift.rerank_vs_vector_only`. `lift.rerank_vs_hybrid` is
reported alongside it because it is the more honest question: it isolates the
reranker's contribution from the hybrid retriever's.

Run:  python -m evals.retrieval.run_retrieval_eval
"""

from __future__ import annotations

import argparse
import sys
import time

from araa.config import Settings, get_settings
from araa.retrieval.hybrid import HybridRetriever
from evals.harness import (
    EvalArtefact,
    environment_stamp,
    fmt_pct,
    hit_at_k,
    mean,
    mrr,
    ndcg_at_k,
    now_iso,
    percentile,
    recall_at_k,
    require_index,
    wilson_interval,
)

CONFIGS: dict[str, dict] = {
    "vector_only": {"use_vector": True, "use_bm25": False, "use_reranker": False},
    "bm25_only": {"use_vector": False, "use_bm25": True, "use_reranker": False},
    "hybrid": {"use_vector": True, "use_bm25": True, "use_reranker": False},
    "hybrid_reranked": {"use_vector": True, "use_bm25": True, "use_reranker": True},
}

K_VALUES = (1, 3, 5, 10)

# The headline metric must match the granularity of the question set's gold, and
# getting this wrong is not a cosmetic problem.
#
# `retrieval_queries` has document-level gold: every body chunk of the source
# paper is marked relevant, ~100 of them. Chunk-level nDCG@10 over a gold set
# that large does not measure ranking quality at all — it rewards returning ten
# chunks from one paper, and *penalises* a reranker for surfacing the single
# best passage alongside diverse context. Measured on this corpus, reranking
# scored -6.2% on chunk_ndcg@10 while simultaneously improving doc_hit@3 from
# 91.7% to 97.2%. The nDCG number was not evidence the reranker hurt; it was
# evidence the metric was wrong for the gold.
#
# `research_questions` has exactly one gold chunk per question, so chunk-level
# ranking metrics are meaningful there and are used instead.
HEADLINE_METRIC: dict[str, str] = {
    "retrieval_queries": "doc_hit@3",
    "research_questions": "chunk_ndcg@10",
}
DEFAULT_HEADLINE = "chunk_ndcg@10"


def evaluate_config(
    retriever: HybridRetriever,
    questions: list[dict],
    flags: dict,
    top_k: int,
) -> dict:
    per_q: list[dict] = []
    latencies: list[float] = []

    for q in questions:
        gold_chunks = set(q.get("gold_chunk_ids") or [])
        gold_docs = set(q.get("gold_doc_ids") or [])
        excluded = set(q.get("exclude_chunk_ids") or [])

        # Retrieve extra so that dropping excluded chunks still leaves top_k.
        t0 = time.perf_counter()
        trace = retriever.retrieve(q["question"], top_k=top_k + len(excluded), **flags)
        latencies.append((time.perf_counter() - t0) * 1000)

        kept = [r for r in trace.results if r.chunk.chunk_id not in excluded][:top_k]
        retrieved_chunks = [r.chunk.chunk_id for r in kept]
        retrieved_docs: list[str] = []
        for r in kept:
            if r.chunk.doc_id not in retrieved_docs:
                retrieved_docs.append(r.chunk.doc_id)

        row = {
            "id": q["id"],
            "question": q["question"][:160],
            "gold_docs": sorted(gold_docs),
            "retrieved_docs": retrieved_docs[:10],
            "chunk_mrr": mrr(retrieved_chunks, gold_chunks),
            "doc_mrr": mrr(retrieved_docs, gold_docs),
        }
        for k in K_VALUES:
            if k > top_k:
                continue
            row[f"chunk_hit@{k}"] = hit_at_k(retrieved_chunks, gold_chunks, k)
            row[f"chunk_recall@{k}"] = recall_at_k(retrieved_chunks, gold_chunks, k)
            row[f"chunk_ndcg@{k}"] = ndcg_at_k(retrieved_chunks, gold_chunks, k)
            row[f"doc_hit@{k}"] = hit_at_k(retrieved_docs, gold_docs, k)
        per_q.append(row)

    metric_keys = [k for k in per_q[0] if isinstance(per_q[0][k], float)] if per_q else []
    metrics = {k: round(mean([r[k] for r in per_q]), 4) for k in metric_keys}
    metrics |= {
        "questions": len(per_q),
        "latency_ms_mean": round(mean(latencies), 1),
        "latency_ms_p95": round(percentile(latencies, 0.95), 1),
    }
    return {"metrics": metrics, "detail": per_q}


def _lift(after: float, before: float, n: int = 0, is_rate: bool = False) -> dict:
    out = {
        "before": round(before, 4),
        "after": round(after, 4),
        "absolute": round(after - before, 4),
        "relative": round((after - before) / before, 4) if before else None,
    }
    if is_rate and n:
        # A hit-rate difference of one or two questions on a set this size is
        # not a result. Quoting the point estimate without the interval invites
        # exactly the over-reading this repo is supposed to avoid.
        lo_b, hi_b = wilson_interval(round(before * n), n)
        lo_a, hi_a = wilson_interval(round(after * n), n)
        out |= {
            "n": n,
            "before_ci95": [round(lo_b, 4), round(hi_b, 4)],
            "after_ci95": [round(lo_a, 4), round(hi_a, 4)],
            "intervals_overlap": lo_a <= hi_b and lo_b <= hi_a,
        }
    return out


def run(
    settings: Settings, question_sets: list[str], top_k: int, detail_limit: int
) -> EvalArtefact:
    from evals.harness import load_question_set

    require_index(settings)
    retriever = HybridRetriever(settings)

    all_results: dict[str, dict] = {}
    notes: list[str] = []

    for set_name in question_sets:
        try:
            questions = load_question_set(set_name)
        except FileNotFoundError as exc:
            notes.append(str(exc).splitlines()[0])
            continue
        if not questions:
            notes.append(f"question set {set_name!r} is empty; skipped")
            continue

        print(f"\n=== {set_name}: {len(questions)} questions ===", file=sys.stderr)
        per_config: dict[str, dict] = {}
        for name, flags in CONFIGS.items():
            if name == "hybrid_reranked" and not settings.use_reranker:
                notes.append("ARAA_USE_RERANKER is false; forcing it on for this configuration")
            print(f"  {name} ...", end="", flush=True, file=sys.stderr)
            out = evaluate_config(retriever, questions, flags, top_k)
            per_config[name] = out
            m = out["metrics"]
            print(
                f" chunk_hit@5={fmt_pct(m.get('chunk_hit@5', 0))}"
                f" doc_hit@3={fmt_pct(m.get('doc_hit@3', 0))}"
                f" ndcg@10={m.get('chunk_ndcg@10', 0):.3f}"
                f" {m['latency_ms_mean']:.0f}ms",
                file=sys.stderr,
            )

        headline = HEADLINE_METRIC.get(set_name, DEFAULT_HEADLINE)
        if headline not in per_config["vector_only"]["metrics"]:
            headline = "chunk_mrr"
            notes.append(
                f"{set_name}: headline metric unavailable at top_k={top_k}; fell back to chunk_mrr"
            )
        # hit@k and recall@k are proportions, so a Wilson interval applies.
        # nDCG and MRR are means of continuous scores and it does not.
        is_rate = headline.startswith(("doc_hit@", "chunk_hit@", "chunk_recall@"))
        n_q = len(questions)
        all_results[set_name] = {
            "questions": len(questions),
            "configs": {
                name: {
                    "metrics": out["metrics"],
                    "detail": out["detail"][:detail_limit],
                }
                for name, out in per_config.items()
            },
            "lift": {
                "metric": headline,
                "rerank_vs_vector_only": _lift(
                    per_config["hybrid_reranked"]["metrics"][headline],
                    per_config["vector_only"]["metrics"][headline],
                    n_q,
                    is_rate,
                ),
                "rerank_vs_hybrid": _lift(
                    per_config["hybrid_reranked"]["metrics"][headline],
                    per_config["hybrid"]["metrics"][headline],
                    n_q,
                    is_rate,
                ),
                "hybrid_vs_vector_only": _lift(
                    per_config["hybrid"]["metrics"][headline],
                    per_config["vector_only"]["metrics"][headline],
                    n_q,
                    is_rate,
                ),
                "rerank_latency_cost_ms": round(
                    per_config["hybrid_reranked"]["metrics"]["latency_ms_mean"]
                    - per_config["hybrid"]["metrics"]["latency_ms_mean"],
                    1,
                ),
            },
        }

    if not all_results:
        raise SystemExit(
            "No question sets available. Build them first:\n"
            "  python -m evals.questions.build_question_set"
        )

    return EvalArtefact(
        name="retrieval",
        metrics=all_results,
        config={
            "top_k": top_k,
            "headline_metrics": HEADLINE_METRIC,
            **settings.retrieval_signature(),
        },
        environment=environment_stamp(settings),
        notes=notes
        + [
            "Lift is reported on the headline metric appropriate to each question "
            "set's gold granularity: doc_hit@3 for the deterministic set (gold is "
            "document-level), chunk_ndcg@10 for the generated set (gold is a single "
            "chunk). Every other metric is recorded per configuration regardless.",
            "The query's own abstract chunks are removed from the retrieved list "
            "before scoring on the deterministic set, since the query is lifted "
            "from that abstract.",
            "Reranker latency is measured on CPU inside a container. A GPU moves "
            "this by roughly an order of magnitude; the quality numbers do not change.",
        ],
        started_at=now_iso(),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--sets",
        nargs="+",
        default=["retrieval_queries", "research_questions"],
        help="question sets to evaluate",
    )
    ap.add_argument(
        "--detail-limit", type=int, default=25, help="per-question rows kept in the artefact"
    )
    args = ap.parse_args()

    s = get_settings()
    artefact = run(s, args.sets, args.top_k, args.detail_limit)
    path = artefact.write(s)

    print("\n" + "=" * 78)
    for set_name, res in artefact.metrics.items():
        lift = res["lift"]
        print(f"\n{set_name}  (n={res['questions']}, headline metric = {lift['metric']})")
        print(
            f"  {'config':<18} {'chunk_hit@5':>12} {'doc_hit@3':>10} {'doc_mrr':>8} {'ndcg@10':>9} {'ms':>7}"
        )
        for name, out in res["configs"].items():
            m = out["metrics"]
            print(
                f"  {name:<18} {fmt_pct(m.get('chunk_hit@5', 0)):>12} "
                f"{fmt_pct(m.get('doc_hit@3', 0)):>10} "
                f"{m.get('doc_mrr', 0):>8.3f} "
                f"{m.get('chunk_ndcg@10', 0):>9.3f} {m['latency_ms_mean']:>7.0f}"
            )
        for label, key in (
            ("hybrid vs vector-only  ", "hybrid_vs_vector_only"),
            ("reranker vs vector-only", "rerank_vs_vector_only"),
            ("reranker vs hybrid     ", "rerank_vs_hybrid"),
        ):
            d = lift[key]
            rel = "n/a" if d["relative"] is None else f"{d['relative']:+.1%}"
            line = (
                f"  {label}: {d['before']:.3f} -> {d['after']:.3f}  ({d['absolute']:+.4f}, {rel})"
            )
            if d.get("intervals_overlap") is True:
                line += f"   [95% CIs overlap - not significant at n={d['n']}]"
            print(line)
        print(f"  reranker latency cost:   +{lift['rerank_latency_cost_ms']:.0f} ms/query (CPU)")

    print(f"\nartefact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

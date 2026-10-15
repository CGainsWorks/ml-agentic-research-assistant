"""Chunking strategy comparison.

Re-ingests the same corpus once per strategy into an isolated index, rebuilds
the deterministic question set against each, and reports retrieval quality plus
the costs each strategy imposes: chunk count, mean chunk length, and the token
budget a query's context actually consumes.

The last of those is why this exists. A strategy that wins recall by emitting
three times as many chunks and stuffing 4x the characters into the context has
not won anything — it has moved the cost somewhere the recall number cannot see.

This is slow (a full re-index per strategy). Run it when the chunking changes,
not on every commit.

Run:  python -m evals.retrieval.run_chunking_comparison
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from araa.config import CHUNKING_GRID, Settings, get_settings
from araa.ingestion.pipeline import run_ingest
from araa.retrieval.hybrid import ChunkStore, HybridRetriever
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
)
from evals.questions.build_question_set import build_retrieval_queries, load_manifest_abstracts


def _settings_for(base: Settings, strategy: str) -> Settings:
    """Isolate each strategy's index so they never contaminate each other."""
    overrides = CHUNKING_GRID.get(strategy, {})
    sandbox = base.data_dir / "chunking_comparison" / strategy
    return base.model_copy(
        update={
            "chunk_strategy": strategy,
            "chroma_dir": sandbox / "chroma",
            "index_dir": sandbox / "index",
            "collection_name": f"{base.collection_name}_{strategy}",
            **overrides,
        }
    )


def _evaluate(settings: Settings, questions: list[dict], top_k: int) -> dict:
    retriever = HybridRetriever(settings)
    rows = []
    latencies: list[float] = []
    context_chars: list[int] = []

    for q in questions:
        gold_chunks = set(q["gold_chunk_ids"])
        gold_docs = set(q["gold_doc_ids"])
        excluded = set(q.get("exclude_chunk_ids") or [])

        t0 = time.perf_counter()
        trace = retriever.retrieve(q["question"], top_k=top_k + len(excluded))
        latencies.append((time.perf_counter() - t0) * 1000)

        kept = [r for r in trace.results if r.chunk.chunk_id not in excluded][:top_k]
        chunk_ids = [r.chunk.chunk_id for r in kept]
        doc_ids: list[str] = []
        for r in kept:
            if r.chunk.doc_id not in doc_ids:
                doc_ids.append(r.chunk.doc_id)

        # What the model would actually be handed for this query.
        context_chars.append(sum(len(r.chunk.text) for r in kept[: settings.final_top_k]))

        rows.append(
            {
                "chunk_hit@5": hit_at_k(chunk_ids, gold_chunks, 5),
                "chunk_hit@10": hit_at_k(chunk_ids, gold_chunks, 10),
                "chunk_ndcg@10": ndcg_at_k(chunk_ids, gold_chunks, 10),
                "chunk_mrr": mrr(chunk_ids, gold_chunks),
                "doc_hit@3": hit_at_k(doc_ids, gold_docs, 3),
            }
        )

    metrics = {k: round(mean([r[k] for r in rows]), 4) for k in rows[0]} if rows else {}
    metrics |= {
        "questions": len(rows),
        "latency_ms_mean": round(mean(latencies), 1),
        "latency_ms_p95": round(percentile(latencies, 0.95), 1),
        "context_chars_mean": round(mean([float(c) for c in context_chars]), 0),
        # ~4 chars/token is the usual English rule of thumb. Approximate, and
        # labelled as such; the exact figure comes from the cost report, which
        # uses the provider's own counts.
        "context_tokens_approx": round(mean([float(c) for c in context_chars]) / 4, 0),
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategies", nargs="+", default=list(CHUNKING_GRID))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--per-doc", type=int, default=1)
    ap.add_argument("--keep-sandboxes", action="store_true", help="do not delete the temp indexes")
    args = ap.parse_args()

    base = get_settings()
    if not (base.corpus_dir / "manifest.json").exists():
        raise SystemExit("No corpus. Run: python scripts/fetch_corpus.py")

    results: dict[str, dict] = {}
    notes: list[str] = []

    for strategy in args.strategies:
        print(f"\n=== {strategy} ===", file=sys.stderr)
        s = _settings_for(base, strategy)
        s.ensure_dirs()

        t0 = time.perf_counter()
        report = run_ingest(s, reset=True)
        ingest_s = time.perf_counter() - t0

        # Question set is rebuilt per strategy: gold chunk ids only exist inside
        # that strategy's own index. The *queries* are identical across
        # strategies because they come from abstracts, not from chunks.
        questions = build_retrieval_queries(
            ChunkStore(s).all(), args.per_doc, base.seed, load_manifest_abstracts(base)
        )
        if not questions:
            notes.append(f"{strategy}: no queries could be built; skipped")
            continue

        metrics = _evaluate(s, questions, args.top_k)
        results[strategy] = {
            "retrieval": metrics,
            "index": {
                "chunks": report.chunks,
                "chunk_chars_mean": report.chunk_chars_mean,
                "chunk_chars_p95": report.chunk_chars_p95,
                "documents": report.documents_ingested,
                "ingest_seconds": round(ingest_s, 1),
            },
            "params": {
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
                "sentence_window_size": s.sentence_window_size,
            },
        }
        print(
            f"  chunks={report.chunks} hit@5={fmt_pct(metrics['chunk_hit@5'])} "
            f"ndcg@10={metrics['chunk_ndcg@10']:.3f} ctx~{metrics['context_tokens_approx']:.0f}tok",
            file=sys.stderr,
        )

        if not args.keep_sandboxes:
            shutil.rmtree(base.data_dir / "chunking_comparison" / strategy, ignore_errors=True)

    if not results:
        raise SystemExit("No strategy produced a usable index.")

    # Primary metric is document-level, because the gold is document-level.
    # Chunk-level nDCG over ~100 gold chunks per query is confounded by how many
    # chunks a strategy emits: sentence_window scored 0.348 while reaching the
    # same doc_hit@3 as everything else, purely because it produces 5020 chunks
    # and the top-5 is therefore spread thinner. Reporting nDCG as the headline
    # would have declared a winner on an artefact of chunk count.
    best_doc = max(results, key=lambda k: results[k]["retrieval"]["doc_hit@3"])
    doc_scores = {k: results[k]["retrieval"]["doc_hit@3"] for k in results}
    tied = [k for k, v in doc_scores.items() if v == doc_scores[best_doc]]
    winner = max(results, key=lambda k: results[k]["retrieval"]["chunk_ndcg@10"])
    cheapest = min(results, key=lambda k: results[k]["retrieval"]["context_tokens_approx"])
    fewest_chunks = min(results, key=lambda k: results[k]["index"]["chunks"])

    artefact = EvalArtefact(
        name="chunking",
        metrics={
            "by_strategy": results,
            "primary_metric": "doc_hit@3",
            "best_doc_hit@3": best_doc,
            "tied_on_doc_hit@3": sorted(tied),
            "separable_on_quality": len(tied) < len(results),
            "best_ndcg@10": winner,
            "lowest_context_cost": cheapest,
            "fewest_chunks": fewest_chunks,
        },
        config={"top_k": args.top_k, "per_doc": args.per_doc, "grid": CHUNKING_GRID},
        environment=environment_stamp(base),
        notes=notes
        + [
            "Each strategy was indexed into its own isolated Chroma collection.",
            "Queries are identical across strategies (abstract sentences); gold chunk "
            "sets are per-strategy because chunk ids differ by construction.",
            "context_tokens_approx is chars/4, not a tokenizer count. The cost report "
            "uses the provider's own token counts.",
            "PRIMARY METRIC IS doc_hit@3. The gold set is document-level, so chunk-level "
            "nDCG@10 is confounded by chunk count: a strategy emitting 2x the chunks "
            "spreads the same relevant material across a longer list and scores worse "
            "without retrieving anything different. Judge quality on doc_hit@3 and cost "
            "on chunks / context_tokens_approx.",
        ],
        started_at=now_iso(),
    )
    path = artefact.write(base)

    print("\n" + "=" * 96)
    print(
        f"  {'strategy':<17}{'chunks':>8}{'mean len':>10}"
        f"{'doc hit@3':>11}{'chunk hit@5':>13}{'ndcg@10':>9}{'ctx tok':>9}{'ms':>7}"
    )
    for name, r in results.items():
        m, ix = r["retrieval"], r["index"]
        print(
            f"  {name:<17}{ix['chunks']:>8}{ix['chunk_chars_mean']:>10.0f}"
            f"{fmt_pct(m['doc_hit@3']):>11}{fmt_pct(m['chunk_hit@5']):>13}"
            f"{m['chunk_ndcg@10']:>9.3f}"
            f"{m['context_tokens_approx']:>9.0f}{m['latency_ms_mean']:>7.0f}"
        )

    print(f"\n  primary metric (doc hit@3): best = {best_doc}")
    if len(tied) == len(results):
        print(
            "  ALL STRATEGIES TIE on doc hit@3. This question set cannot separate them\n"
            "  on retrieval quality; choose on cost instead. The chunk-level columns\n"
            "  differ mainly because the strategies emit different numbers of chunks."
        )
    elif len(tied) > 1:
        print(f"  tied on doc hit@3: {', '.join(sorted(tied))}")
    print(f"  lowest context cost: {cheapest}     fewest chunks: {fewest_chunks}")
    print(f"\nartefact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

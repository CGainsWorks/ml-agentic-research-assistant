"""Hybrid retrieval: vector + keyword, fused, then reranked.

    query
      |-- vector (bge-small, cosine, top 30) ----.
      |                                          |--> RRF fuse (top 20) --> cross-encoder --> top 6
      '-- keyword (BM25, top 30) ---------------'

Reciprocal Rank Fusion rather than a weighted score blend, because cosine
similarity and BM25 scores are not on comparable scales and any fixed weight
between them is a magic number that needs retuning every time the corpus
changes. RRF only uses rank position, so it needs no tuning and no calibration:

    score(d) = sum over retrievers of  1 / (k + rank(d))

The pipeline emits a full trace at every stage — that trace is what the API
returns as `retrieval_trace` and what the retrieval eval measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from araa.config import Settings
from araa.ingestion.documents import Chunk, read_chunks_jsonl
from araa.retrieval.bm25 import BM25Index
from araa.retrieval.embed import Embedder
from araa.retrieval.rerank import Reranker
from araa.retrieval.vectorstore import VectorStore


@dataclass(slots=True)
class Retrieved:
    chunk: Chunk
    score: float
    rank: int
    # Per-stage bookkeeping, so a trace can answer "why is this chunk here".
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self, include_text: bool = True) -> dict:
        d = {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.metadata.get("title"),
            "section": self.chunk.metadata.get("section"),
            "page": self.chunk.metadata.get("page"),
            "licence": self.chunk.metadata.get("licence"),
            "rank": self.rank,
            "score": round(self.score, 6),
            "vector_rank": self.vector_rank,
            "vector_score": None if self.vector_score is None else round(self.vector_score, 6),
            "bm25_rank": self.bm25_rank,
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 6),
            "rrf_score": None if self.rrf_score is None else round(self.rrf_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }
        if include_text:
            d["text"] = self.chunk.text
        return d


@dataclass
class RetrievalTrace:
    query: str
    vector_hits: int = 0
    bm25_hits: int = 0
    fused_candidates: int = 0
    reranked: bool = False
    results: list[Retrieved] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_text: bool = False) -> dict:
        return {
            "query": self.query,
            "vector_hits": self.vector_hits,
            "bm25_hits": self.bm25_hits,
            "fused_candidates": self.fused_candidates,
            "reranked": self.reranked,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "results": [r.to_dict(include_text) for r in self.results],
        }


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]], k: int) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class ChunkStore:
    """chunk_id -> Chunk, loaded from the ingestion artefact."""

    def __init__(self, settings: Settings) -> None:
        self.path: Path = settings.index_dir / "chunks.jsonl"
        self._by_id: dict[str, Chunk] = {}
        self._loaded = False

    def load(self) -> int:
        chunks = read_chunks_jsonl(self.path)
        self._by_id = {c.chunk_id: c for c in chunks}
        self._loaded = True
        return len(self._by_id)

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, chunk_id: str) -> Chunk | None:
        self.ensure_loaded()
        return self._by_id.get(chunk_id)

    def all(self) -> list[Chunk]:
        self.ensure_loaded()
        return list(self._by_id.values())

    def __len__(self) -> int:
        self.ensure_loaded()
        return len(self._by_id)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        chunks: ChunkStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self.vector = vector_store or VectorStore(settings, self.embedder)
        self.bm25 = bm25 or BM25Index(settings)
        self.chunks = chunks or ChunkStore(settings)
        self._reranker = reranker
        self._rerank_requested = settings.use_reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.settings)
        return self._reranker

    def is_ready(self) -> bool:
        try:
            return self.vector.count() > 0 and len(self.chunks) > 0
        except Exception:  # noqa: BLE001
            return False

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_reranker: bool | None = None,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> RetrievalTrace:
        import time

        s = self.settings
        top_k = top_k or s.final_top_k
        rerank_on = self._rerank_requested if use_reranker is None else use_reranker

        trace = RetrievalTrace(query=query)

        t0 = time.perf_counter()
        vector_hits = self.vector.query(query, s.vector_top_k) if use_vector else []
        trace.timings_ms["vector"] = (time.perf_counter() - t0) * 1000
        trace.vector_hits = len(vector_hits)

        t0 = time.perf_counter()
        bm25_hits = self.bm25.query(query, s.bm25_top_k) if use_bm25 else []
        trace.timings_ms["bm25"] = (time.perf_counter() - t0) * 1000
        trace.bm25_hits = len(bm25_hits)

        vector_pos = {cid: (i + 1, sc) for i, (cid, sc) in enumerate(vector_hits)}
        bm25_pos = {cid: (i + 1, sc) for i, (cid, sc) in enumerate(bm25_hits)}

        rankings = [r for r in (vector_hits, bm25_hits) if r]
        if not rankings:
            return trace

        if len(rankings) == 1:
            # Single-retriever mode (used by the ablations in the retrieval eval):
            # keep the retriever's own ordering rather than fusing with nothing.
            fused = {cid: 1.0 / (s.rrf_k + i + 1) for i, (cid, _) in enumerate(rankings[0])}
        else:
            fused = reciprocal_rank_fusion(rankings, s.rrf_k)

        candidates = sorted(fused.items(), key=lambda kv: -kv[1])[: s.fused_top_k]
        trace.fused_candidates = len(candidates)

        resolved: list[tuple[str, Chunk, float]] = []
        for cid, rrf in candidates:
            chunk = self.chunks.get(cid)
            if chunk is not None:
                resolved.append((cid, chunk, rrf))

        if rerank_on and resolved:
            t0 = time.perf_counter()
            scored = self.reranker.rerank(
                query, [(cid, chunk.text) for cid, chunk, _ in resolved], top_k=top_k
            )
            trace.timings_ms["rerank"] = (time.perf_counter() - t0) * 1000
            trace.reranked = True
            rrf_by_id = {cid: rrf for cid, _, rrf in resolved}
            chunk_by_id = {cid: c for cid, c, _ in resolved}
            ordered = [(cid, chunk_by_id[cid], rrf_by_id[cid], score) for cid, score in scored]
        else:
            ordered = [(cid, chunk, rrf, None) for cid, chunk, rrf in resolved[:top_k]]

        results: list[Retrieved] = []
        for rank, (cid, chunk, rrf, rr_score) in enumerate(ordered, start=1):
            v = vector_pos.get(cid)
            b = bm25_pos.get(cid)
            results.append(
                Retrieved(
                    chunk=chunk,
                    score=rr_score if rr_score is not None else rrf,
                    rank=rank,
                    vector_rank=v[0] if v else None,
                    vector_score=v[1] if v else None,
                    bm25_rank=b[0] if b else None,
                    bm25_score=b[1] if b else None,
                    rrf_score=rrf,
                    rerank_score=rr_score,
                )
            )
        trace.results = results
        return trace

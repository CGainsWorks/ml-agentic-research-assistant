"""Cross-encoder reranking.

The bi-encoder used for the vector index embeds query and document separately —
fast, but it never gets to compare them token against token. A cross-encoder
reads the pair jointly and scores relevance directly. It is far too slow to run
over the whole corpus, which is exactly why it sits at the end of the pipeline
scoring ~20 fused candidates rather than ~20,000 chunks.

Whether it is worth the latency is not a matter of opinion here:
`evals/retrieval/run_retrieval_eval.py` reports the measured lift over the
vector-only baseline, and `ARAA_USE_RERANKER=false` turns it off.
"""

from __future__ import annotations

import functools
import threading

from araa.config import Settings

_LOCK = threading.Lock()


@functools.lru_cache(maxsize=2)
def _load_cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device="cpu", max_length=512)


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            with _LOCK:
                if self._model is None:
                    self._model = _load_cross_encoder(self.settings.reranker_model)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(
            pairs,
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def rerank(
        self, query: str, items: list[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        """items: (id, text). Returns (id, score) sorted descending."""
        if not items:
            return []
        ids = [i for i, _ in items]
        scores = self.score(query, [t for _, t in items])
        ranked = sorted(zip(ids, scores, strict=True), key=lambda kv: -kv[1])
        return ranked[:top_k]

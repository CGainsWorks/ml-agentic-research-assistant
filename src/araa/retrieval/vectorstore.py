"""ChromaDB vector store.

Chroma over Pinecone because the brief allows either and only one of them runs
on a laptop with no account, which is what "everything runs from docker compose
up" requires.

Embeddings are computed by our own `Embedder` and passed in explicitly rather
than letting Chroma pick a default embedding function — otherwise the vector
index and the reranker would silently disagree about what model produced what,
and the eval numbers would not be attributable to a named model.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from araa.config import Settings
from araa.ingestion.documents import Chunk
from araa.retrieval.embed import Embedder, texts_for_embedding

# Chroma 0.5.x calls its telemetry hook with the wrong arity and logs the
# resulting TypeError once per operation, even with telemetry disabled. It is
# harmless and it is not ours, but a hundred lines of it buries the output of
# every ingest and eval run. Set before chromadb is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

_SCALAR = (str, int, float, bool)


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma only stores scalar metadata; anything else is JSON-encoded."""
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, _SCALAR):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


class VectorStore:
    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
        )

    # ------------------------------------------------------------- writing
    def reset(self) -> None:
        try:
            self._client.delete_collection(self.settings.collection_name)
        except Exception:  # noqa: BLE001 - absent collection is fine
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.settings.embedding_model},
        )

    def add(self, chunks: list[Chunk], batch_size: int = 256) -> int:
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self.embedder.embed_documents(texts_for_embedding(batch))
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[v.tolist() for v in vectors],
                documents=[c.text for c in batch],
                metadatas=[_flatten_metadata(c.metadata | {"ordinal": c.ordinal}) for c in batch],
            )
            added += len(batch)
        return added

    # ------------------------------------------------------------- reading
    def count(self) -> int:
        return int(self._collection.count())

    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, similarity) descending. Similarity is 1 - cosine distance."""
        if self.count() == 0:
            return []
        vector = self.embedder.embed_query(query)
        res = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, self.count()),
            include=["distances"],
        )
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists, strict=False)]

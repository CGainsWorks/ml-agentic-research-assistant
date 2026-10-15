"""Embeddings.

BAAI/bge-small-en-v1.5: 384 dims, 33M params, MIT licensed, runs on CPU in the
container. Chosen over an API embedding model so that the retrieval numbers in
the README are reproducible by anyone with no key and no spend — which is the
entire point of the eval harness.

Two bge-specific details that are easy to get wrong and cost real recall:

* The query side wants an instruction prefix; the document side must not have
  one. Embedding both sides identically measurably degrades retrieval.
* The model is trained for cosine similarity, so embeddings are L2-normalised
  here and the vector store is configured for cosine distance.

`sentence_window` chunking stores a different string for embedding than for
reading. `texts_for_embedding` honours that; everything else is unaware of it.
"""

from __future__ import annotations

import functools
import threading

import numpy as np

from araa.config import Settings
from araa.ingestion.documents import Chunk

_LOCK = threading.Lock()


@functools.lru_cache(maxsize=4)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            with _LOCK:
                if self._model is None:
                    self._model = _load_model(self.settings.embedding_model)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        prefixed = f"{self.settings.embedding_query_prefix}{query}"
        return self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]


def texts_for_embedding(chunks: list[Chunk]) -> list[str]:
    """What actually goes into the index.

    sentence_window chunks embed a single sentence but are read back with their
    surrounding context; every other strategy embeds what it reads.
    """
    return [c.metadata.get("embed_text") or c.text for c in chunks]

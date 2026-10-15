"""BM25 keyword retrieval - the other half of "hybrid".

Vector search is bad at exactly the queries a research corpus attracts: exact
model names, dataset names, metric names, acronyms, arXiv ids. "BLEU" and
"ROUGE" are near neighbours in embedding space and completely different in
meaning. BM25 gets those right because it matches the literal token.

The index is a plain JSON sidecar rather than a pickle so it can be inspected,
diffed, and read by a future version without a compatible Python.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from araa.config import Settings
from araa.ingestion.documents import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.]*")

# Kept deliberately short. Aggressive stopword removal hurts on technical text
# where "no", "not", and "all" are load-bearing in method names.
_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "for",
    "on",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "with",
    "that",
    "this",
    "these",
    "those",
    "as",
    "at",
    "by",
    "from",
    "it",
    "its",
    "we",
    "our",
    "can",
    "which",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) == 1:
            continue
        tokens.append(raw)
        # Split hyphenated/dotted compounds too, so "self-attention" also
        # matches a query for "attention".
        if "-" in raw or "." in raw:
            tokens.extend(p for p in re.split(r"[-.]", raw) if len(p) > 1 and p not in _STOPWORDS)
    return tokens


class BM25Index:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path: Path = settings.index_dir / "bm25.json"
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------- writing
    def build(self, chunks: list[Chunk]) -> int:
        self._ids = [c.chunk_id for c in chunks]
        corpus = [tokenize(_indexable_text(c)) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"ids": self._ids, "corpus": corpus}, ensure_ascii=False),
            encoding="utf-8",
        )
        return len(self._ids)

    # ------------------------------------------------------------- reading
    def load(self) -> bool:
        if not self.path.exists():
            return False
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._ids = data["ids"]
        corpus = data["corpus"]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        return self._bm25 is not None

    @property
    def size(self) -> int:
        return len(self._ids)

    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None and not self.load():
            return []
        assert self._bm25 is not None
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda kv: -kv[1])
        return [(cid, float(s)) for cid, s in ranked[:top_k] if s > 0.0]


def _indexable_text(chunk: Chunk) -> str:
    """Title and section go into the BM25 field.

    A chunk from the middle of a paper often never repeats the paper's own
    subject; folding the title in is what lets a keyword query for a model name
    reach the paragraph that discusses it without naming it again.
    """
    bits = [chunk.text]
    if title := chunk.metadata.get("title"):
        bits.append(str(title))
    if section := chunk.metadata.get("section"):
        bits.append(str(section))
    return "\n".join(bits)

"""Document and chunk data model.

Parsers produce `SourceDocument`s made of `Block`s. Chunkers consume `Block`s and
produce `Chunk`s. Keeping the two apart is what makes the chunking strategy
swappable: a new strategy is a function over blocks, not a new parser.

A `Block` carries page and section provenance so that a citation can point at
"page 4, section 3.2 of arXiv:2401.12345" rather than at an opaque chunk id.
That provenance is what the groundedness eval checks against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

BlockType = Literal["title", "abstract", "body", "table", "record"]


@dataclass(slots=True)
class Block:
    text: str
    block_type: BlockType = "body"
    page: int | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()


@dataclass(slots=True)
class SourceDocument:
    doc_id: str
    title: str
    blocks: list[Block]
    source_path: str | None = None
    # Licence is mandatory. A document with no recorded licence never reaches
    # the index; see ingestion/pipeline.py.
    licence: str = ""
    licence_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, ordinal: int, text: str) -> str:
        digest = hashlib.sha1(f"{doc_id}:{ordinal}:{text}".encode()).hexdigest()[:12]
        return f"{doc_id}#{ordinal:04d}-{digest}"

    def citation_label(self) -> str:
        """Human-readable provenance used in prompts and in the final answer."""
        parts = [self.metadata.get("title") or self.doc_id]
        if self.metadata.get("section"):
            parts.append(f"§{self.metadata['section']}")
        if self.metadata.get("page") is not None:
            parts.append(f"p.{self.metadata['page']}")
        return ", ".join(str(p) for p in parts)


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Chunk(**json.loads(line)))
    return out

"""Chunking strategies, swappable and comparable.

All four implement the same signature so `evals/retrieval/run_chunking_comparison.py`
can hold everything else fixed and vary only this. That comparison is the point:
"we chose recursive chunking" is an opinion, "recursive beat fixed by N points of
recall@10 on this question set" is a result.

    fixed            character windows, no overlap. The naive baseline.
    recursive        character windows that prefer to break at paragraph, then
                     sentence, then word boundaries, with overlap.
    sentence_window  embed one sentence, retrieve it with N sentences of context
                     either side. Precise retrieval, generous context.
    section_aware    respect the document's own section structure; split a
                     section only when it exceeds the size budget.

Tables are never split by any strategy. Half a markdown table retrieves badly
and reads worse.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from araa.config import Settings
from araa.ingestion.documents import Block, Chunk, SourceDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _chunk_meta(doc: SourceDocument, block: Block, extra: dict | None = None) -> dict:
    meta = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "licence": doc.licence,
        "licence_url": doc.licence_url,
        "page": block.page,
        "section": block.section,
        "block_type": block.block_type,
    }
    meta.update(doc.metadata)
    if extra:
        meta.update(extra)
    return meta


def _emit(
    doc: SourceDocument, out: list[Chunk], text: str, block: Block, extra: dict | None = None
):
    text = text.strip()
    if len(text) < 30:
        return
    ordinal = len(out)
    out.append(
        Chunk(
            chunk_id=Chunk.make_id(doc.doc_id, ordinal, text),
            doc_id=doc.doc_id,
            text=text,
            ordinal=ordinal,
            metadata=_chunk_meta(doc, block, extra),
        )
    )


# --------------------------------------------------------------------- fixed
def chunk_fixed(doc: SourceDocument, settings: Settings) -> list[Chunk]:
    size = settings.chunk_size
    out: list[Chunk] = []
    for block in doc.blocks:
        if block.block_type == "table":
            _emit(doc, out, block.text, block)
            continue
        for i in range(0, len(block.text), size):
            _emit(doc, out, block.text[i : i + size], block)
    return out


# ----------------------------------------------------------------- recursive
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = seps[0], seps[1:]
    pieces = text.split(sep)
    out: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}{sep}{piece}" if buf else piece
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(piece) > size:
            out.extend(_recursive_split(piece, size, rest))
            buf = ""
        else:
            buf = piece
    if buf:
        out.append(buf)
    return [p for p in out if p.strip()]


def chunk_recursive(doc: SourceDocument, settings: Settings) -> list[Chunk]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    out: list[Chunk] = []
    for block in doc.blocks:
        if block.block_type == "table":
            _emit(doc, out, block.text, block)
            continue
        pieces = _recursive_split(block.text, size, _SEPARATORS)
        for i, piece in enumerate(pieces):
            # Prepend the tail of the previous piece so a sentence split across
            # a boundary is still retrievable from either side.
            prefix = pieces[i - 1][-overlap:] if (i > 0 and overlap) else ""
            _emit(doc, out, f"{prefix}{piece}" if prefix else piece, block)
    return out


# ----------------------------------------------------- sentence with a window
def chunk_sentence_window(doc: SourceDocument, settings: Settings) -> list[Chunk]:
    """Embed one sentence, store its neighbourhood as the text handed to the LLM.

    `embed_text` is what goes into the vector index; `text` is what the model
    reads. The retrieval layer honours this split — see retrieval/embed.py.
    """
    window = settings.sentence_window_size
    out: list[Chunk] = []
    for block in doc.blocks:
        if block.block_type == "table":
            _emit(doc, out, block.text, block)
            continue
        sentences = split_sentences(block.text)
        for i, sent in enumerate(sentences):
            lo, hi = max(0, i - window), min(len(sentences), i + window + 1)
            context = " ".join(sentences[lo:hi])
            if len(sent.strip()) < 30:
                continue
            ordinal = len(out)
            out.append(
                Chunk(
                    chunk_id=Chunk.make_id(doc.doc_id, ordinal, sent),
                    doc_id=doc.doc_id,
                    text=context,
                    ordinal=ordinal,
                    metadata=_chunk_meta(doc, block, {"embed_text": sent}),
                )
            )
    return out


# ------------------------------------------------------------- section aware
def chunk_section_aware(doc: SourceDocument, settings: Settings) -> list[Chunk]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    out: list[Chunk] = []

    groups: list[tuple[Block, list[str]]] = []
    for block in doc.blocks:
        if block.block_type == "table":
            groups.append((block, [block.text]))
            continue
        if (
            groups
            and groups[-1][0].section == block.section
            and groups[-1][0].block_type != "table"
        ):
            groups[-1][1].append(block.text)
        else:
            groups.append((block, [block.text]))

    for head_block, texts in groups:
        merged = "\n\n".join(texts)
        if head_block.block_type == "table" or len(merged) <= size:
            _emit(doc, out, merged, head_block)
            continue
        pieces = _recursive_split(merged, size, _SEPARATORS)
        for i, piece in enumerate(pieces):
            prefix = pieces[i - 1][-overlap:] if (i > 0 and overlap) else ""
            _emit(
                doc,
                out,
                f"{prefix}{piece}" if prefix else piece,
                head_block,
                {"section_part": f"{i + 1}/{len(pieces)}"},
            )
    return out


STRATEGIES: dict[str, Callable[[SourceDocument, Settings], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "sentence_window": chunk_sentence_window,
    "section_aware": chunk_section_aware,
}


def chunk_document(doc: SourceDocument, settings: Settings) -> list[Chunk]:
    try:
        fn = STRATEGIES[settings.chunk_strategy]
    except KeyError:
        raise ValueError(
            f"unknown chunk_strategy {settings.chunk_strategy!r}; "
            f"available: {sorted(STRATEGIES)}"
        ) from None
    return fn(doc, settings)

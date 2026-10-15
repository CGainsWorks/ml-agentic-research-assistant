"""Citation formatting, parsing and validation.

The contract between the prompt and the parser:

* Context passages are presented to the model numbered `[C1]`, `[C2]`, ...
* The model must end every factual sentence with one or more of those markers.
* Anything the model writes that carries no marker is treated as an uncited
  claim, not as a harmless aside.

Validation is mechanical and happens before any LLM judge sees the answer:

* a marker that does not exist in the supplied context is a **fabricated
  citation** — the strongest possible signal, and it needs no judge to detect;
* a factual sentence with no marker is an **uncited claim**;
* both are surfaced in the API response and counted by the groundedness eval.

Sentences that make no factual claim (hedges, transitions, headings, the
"I could not find..." refusal) are excluded from the denominator, because
demanding a citation on "Here is what I found." would make the metric measure
prose style rather than grounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from araa.ingestion.chunking import split_sentences
from araa.ingestion.documents import Chunk
from araa.retrieval.hybrid import Retrieved

MARKER_RE = re.compile(r"\[C(\d+)\]")

# Sentences that assert nothing and so cannot be grounded or ungrounded.
_NON_FACTUAL = re.compile(
    r"^\s*(?:"
    r"#{1,6}\s"  # markdown heading
    r"|[-*+]\s*$"  # empty bullet
    r"|(here (is|are)|below (is|are)|in summary|to summari[sz]e|overall)\b"
    r"|(the (retrieved|available|provided) (context|passages|sources)\b)"
    r"|(i (could not|couldn't|cannot|can't) find\b)"
    r"|(no (relevant )?(information|evidence|passages?|sources?) (was|were) found)"
    r")",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Claim:
    text: str
    markers: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    unknown_markers: list[int] = field(default_factory=list)
    factual: bool = True

    @property
    def is_cited(self) -> bool:
        return bool(self.chunk_ids)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "markers": [f"C{m}" for m in self.markers],
            "chunk_ids": self.chunk_ids,
            "unknown_markers": [f"C{m}" for m in self.unknown_markers],
            "factual": self.factual,
            "cited": self.is_cited,
        }


@dataclass(slots=True)
class CitationReport:
    claims: list[Claim]
    total_claims: int
    cited_claims: int
    uncited_claims: int
    fabricated_markers: list[str]
    used_chunk_ids: list[str]

    @property
    def citation_rate(self) -> float:
        return self.cited_claims / self.total_claims if self.total_claims else 0.0

    def to_dict(self) -> dict:
        return {
            "total_claims": self.total_claims,
            "cited_claims": self.cited_claims,
            "uncited_claims": self.uncited_claims,
            "citation_rate": round(self.citation_rate, 4),
            "fabricated_markers": self.fabricated_markers,
            "used_chunk_ids": self.used_chunk_ids,
            "claims": [c.to_dict() for c in self.claims],
        }


def format_context(
    results: list[Retrieved], max_chars_per_chunk: int | None = None
) -> tuple[str, dict[int, Chunk]]:
    """Render retrieved chunks as a numbered context block.

    Provenance goes in the header of each passage rather than only in a
    post-hoc mapping, so the model can attribute a claim to "the LoRA paper,
    section 4" while it writes, instead of only to an opaque number.
    """
    if max_chars_per_chunk is None:
        from araa.config import get_settings

        max_chars_per_chunk = get_settings().max_context_chars_per_chunk

    lines: list[str] = []
    id_map: dict[int, Chunk] = {}
    for i, r in enumerate(results, start=1):
        chunk = r.chunk
        id_map[i] = chunk
        meta = chunk.metadata
        header_bits = [str(meta.get("title") or chunk.doc_id)]
        if meta.get("section"):
            header_bits.append(f"section {meta['section']}")
        if meta.get("page") is not None:
            header_bits.append(f"page {meta['page']}")
        text = chunk.text
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rsplit(" ", 1)[0] + " ..."
        lines.append(f"[C{i}] ({' | '.join(header_bits)})\n{text}")
    return "\n\n".join(lines), id_map


def is_factual(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 25:
        return False
    if _NON_FACTUAL.match(s):
        return False
    # A line that is only citation markers is not a claim.
    return bool(MARKER_RE.sub("", s).strip())


def parse_citations(answer: str, id_map: dict[int, Chunk]) -> CitationReport:
    claims: list[Claim] = []
    fabricated: set[int] = set()
    used: list[str] = []

    for raw in _split_claims(answer):
        markers = [int(m) for m in MARKER_RE.findall(raw)]
        known = [m for m in markers if m in id_map]
        unknown = [m for m in markers if m not in id_map]
        fabricated.update(unknown)

        chunk_ids = []
        for m in known:
            cid = id_map[m].chunk_id
            chunk_ids.append(cid)
            if cid not in used:
                used.append(cid)

        claims.append(
            Claim(
                text=raw.strip(),
                markers=markers,
                chunk_ids=chunk_ids,
                unknown_markers=unknown,
                factual=is_factual(raw),
            )
        )

    factual_claims = [c for c in claims if c.factual]
    cited = sum(1 for c in factual_claims if c.is_cited)
    return CitationReport(
        claims=claims,
        total_claims=len(factual_claims),
        cited_claims=cited,
        uncited_claims=len(factual_claims) - cited,
        fabricated_markers=[f"C{m}" for m in sorted(fabricated)],
        used_chunk_ids=used,
    )


def _split_claims(answer: str) -> list[str]:
    """Sentences, but respecting markdown bullets and numbered lists.

    A bulleted answer is one 'sentence' per bullet as far as citation is
    concerned; splitting a bullet mid-way would orphan its marker.
    """
    out: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+)", stripped):
            out.append(stripped)
        else:
            out.extend(split_sentences(stripped))
    return [s for s in out if s.strip()]


def render_sources(used_chunk_ids: list[str], id_map: dict[int, Chunk]) -> list[dict]:
    """The source list appended to an answer, deduplicated by document."""
    by_chunk = {c.chunk_id: c for c in id_map.values()}
    seen_docs: set[str] = set()
    sources: list[dict] = []
    for cid in used_chunk_ids:
        chunk = by_chunk.get(cid)
        if chunk is None or chunk.doc_id in seen_docs:
            continue
        seen_docs.add(chunk.doc_id)
        meta = chunk.metadata
        sources.append(
            {
                "doc_id": chunk.doc_id,
                "title": meta.get("title"),
                "url": meta.get("url"),
                "licence": meta.get("licence"),
                "licence_url": meta.get("licence_url"),
                "authors": meta.get("authors"),
            }
        )
    return sources

"""PDF parsing.

arXiv PDFs are two-column LaTeX output with figures, tables, equations and a
reference list. Three things matter for retrieval quality and none of them are
free:

1. Reading order. PyMuPDF's default text extraction reads left-to-right across
   the page, which interleaves the two columns into nonsense. We sort blocks
   into columns first.
2. Section headings. Retrieval provenance is much more useful when a chunk
   knows it came from "3.2 Experimental setup", and section-aware chunking
   depends on it entirely.
3. Tables. Extracted separately and rendered as markdown, because a table
   flattened into prose is unretrievable and unreadable.

The reference list is dropped: it is ~15% of the token budget and answers no
question the corpus is being asked.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from araa.ingestion.documents import Block, SourceDocument

# "3.2 Experimental Setup", "IV. RESULTS", "Appendix A"
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})"  # 3.2 Title Case Heading
    r"|((?:[IVXLC]+)\.)\s+([A-Z][A-Z \-]{2,80})"  # IV. UPPER CASE HEADING
    r"|(Appendix\s+[A-Z])\b[.:]?\s*([^\n]{0,80})"  # Appendix A
    r")\s*$"
)

_REFERENCES_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)

_ABSTRACT_RE = re.compile(r"^\s*abstract\b[.:—-]?\s*", re.IGNORECASE)


def _column_sorted_blocks(page: fitz.Page) -> list[tuple[float, float, str]]:
    """Return (x0, y0, text) blocks in true reading order.

    Splits the page at its horizontal midpoint and reads the left column top to
    bottom, then the right column. Single-column pages fall out of this
    naturally because every block lands in the left bucket.
    """
    raw = page.get_text("blocks")
    page_mid = page.rect.width / 2

    left: list[tuple[float, float, str]] = []
    right: list[tuple[float, float, str]] = []
    for x0, y0, x1, _y1, text, _bno, btype in raw:
        if btype != 0 or not text.strip():  # 0 == text block
            continue
        # A block that straddles the midline is full-width (title, wide table):
        # keep it in the left bucket so it sorts by y with everything above it.
        if x1 <= page_mid + 8:
            left.append((x0, y0, text))
        elif x0 >= page_mid - 8:
            right.append((x0, y0, text))
        else:
            left.append((x0, y0, text))

    left.sort(key=lambda b: b[1])
    right.sort(key=lambda b: b[1])
    return left + right


def _classify_heading(line: str) -> str | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    groups = [g for g in m.groups() if g]
    return " ".join(g.strip() for g in groups).strip()


def parse_pdf(
    path: Path,
    *,
    doc_id: str,
    title: str,
    licence: str,
    licence_url: str,
    metadata: dict | None = None,
    drop_references: bool = True,
    max_pages: int = 60,
) -> SourceDocument:
    doc = fitz.open(path)
    blocks: list[Block] = []
    current_section: str | None = None
    hit_references = False

    try:
        for page_no, page in enumerate(doc, start=1):
            if page_no > max_pages:
                break

            # --- tables first, so their text can be excluded from prose ----
            table_rects = []
            try:
                tables = page.find_tables()
            except Exception:  # pragma: no cover - PyMuPDF table finder is best-effort
                tables = []
            for tbl in tables:
                try:
                    md = _table_to_markdown(tbl.extract())
                except Exception:  # pragma: no cover
                    continue
                if md:
                    table_rects.append(fitz.Rect(tbl.bbox))
                    blocks.append(
                        Block(
                            text=md,
                            block_type="table",
                            page=page_no,
                            section=current_section,
                        )
                    )

            # --- prose ------------------------------------------------------
            for x0, y0, text in _column_sorted_blocks(page):
                if any(r.contains(fitz.Point(x0 + 2, y0 + 2)) for r in table_rects):
                    continue  # already captured as a table

                cleaned = _clean(text)
                if not cleaned:
                    continue

                for para in cleaned.split("\n\n"):
                    para = para.strip()
                    if not para:
                        continue

                    first_line = para.split("\n", 1)[0]
                    if _REFERENCES_RE.match(first_line):
                        hit_references = True
                        if drop_references:
                            break
                    if hit_references and drop_references:
                        break

                    heading = _classify_heading(first_line)
                    if heading and len(para.split("\n")) == 1:
                        current_section = heading
                        continue

                    block_type = "body"
                    if _ABSTRACT_RE.match(para):
                        # Two layouts: "Abstract" alone on its own line (most
                        # arXiv preprints) and "Abstract—<text>" inline (IEEE).
                        # Strip the label; whatever is left is abstract text.
                        block_type = "abstract"
                        para = _ABSTRACT_RE.sub("", para, count=1).strip()
                        current_section = "Abstract"
                        if not para:
                            continue  # bare label, nothing to emit
                    elif current_section == "Abstract":
                        # Paragraphs following the label are still the abstract,
                        # until a numbered heading ends the section. Without this
                        # the abstract is indistinguishable from body text, and
                        # the abstract is the single most useful block a paper
                        # has for both retrieval and eval-set construction.
                        block_type = "abstract"

                    if len(para) < 40 and block_type == "body":
                        continue  # page furniture, figure caption fragments, page numbers

                    blocks.append(
                        Block(
                            text=para,
                            block_type=block_type,
                            page=page_no,
                            section=current_section,
                        )
                    )
                if hit_references and drop_references:
                    break
            if hit_references and drop_references:
                break
    finally:
        doc.close()

    return SourceDocument(
        doc_id=doc_id,
        title=title,
        blocks=blocks,
        source_path=str(path),
        licence=licence,
        licence_url=licence_url,
        metadata=metadata or {},
    )


def _clean(text: str) -> str:
    # De-hyphenate words split across line breaks: "convolu-\ntional" -> "convolutional"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Single newlines inside a paragraph are line wraps, not paragraph breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Ligatures PyMuPDF sometimes leaves in.
    for a, b in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl")):
        text = text.replace(a, b)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in rows if row]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)

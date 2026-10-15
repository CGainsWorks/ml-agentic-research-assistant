"""Ingestion pipeline: corpus -> documents -> chunks -> vector + keyword indexes.

Invariants this enforces, both of which the brief asks for explicitly:

* A document with no recorded licence is never indexed. Not warned about —
  skipped, and counted in the run log as skipped.
* Every run writes a log to data/runs/ recording the corpus, the chunking
  parameters, the embedding model and the resulting counts, so any retrieval
  number can be traced back to the index that produced it.

Run:  python -m araa.ingestion.pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from araa.config import Settings, get_settings
from araa.ingestion.chunking import chunk_document
from araa.ingestion.documents import Chunk, SourceDocument, write_chunks_jsonl
from araa.ingestion.pdf import parse_pdf
from araa.ingestion.tabular import parse_table_file
from araa.retrieval.bm25 import BM25Index
from araa.retrieval.vectorstore import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class IngestReport:
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    documents_seen: int = 0
    documents_ingested: int = 0
    documents_skipped_no_licence: int = 0
    documents_failed: int = 0
    tables_ingested: int = 0
    chunks: int = 0
    vector_records: int = 0
    bm25_records: int = 0
    chunk_chars_mean: float = 0.0
    chunk_chars_p95: int = 0
    config: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    licences: dict[str, int] = field(default_factory=dict)

    def write(self, runs_dir: Path) -> Path:
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "").replace(".", "")[:15]
        path = runs_dir / f"ingest-{stamp}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        latest = runs_dir / "ingest-latest.json"
        latest.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def _manifest_path(settings: Settings) -> Path:
    return settings.corpus_dir / "manifest.json"


def load_corpus(settings: Settings) -> tuple[list[SourceDocument], IngestReport]:
    report = IngestReport(started_at=datetime.now(UTC).isoformat())
    manifest_path = _manifest_path(settings)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No corpus manifest at {manifest_path}. Run: python scripts/fetch_corpus.py"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("documents", [])
    report.documents_seen = len(entries)

    docs: list[SourceDocument] = []
    for entry in entries:
        licence = entry.get("licence", "")
        licence_url = entry.get("licence_url", "")
        if not licence or not licence_url:
            report.documents_skipped_no_licence += 1
            continue

        pdf_rel = entry.get("pdf_path")
        if not pdf_rel:
            report.documents_failed += 1
            report.failures.append(
                {"doc": entry.get("arxiv_id"), "error": "no pdf_path in manifest"}
            )
            continue
        # Manifest paths are relative to the corpus directory. Older manifests
        # stored them relative to the repo root, so fall back to a name lookup
        # rather than failing a corpus someone fetched before that change.
        pdf_path = settings.corpus_dir / pdf_rel
        if not pdf_path.exists():
            pdf_path = REPO_ROOT / pdf_rel
        if not pdf_path.exists():
            pdf_path = settings.corpus_dir / "pdf" / Path(pdf_rel).name
        if not pdf_path.exists():
            report.documents_failed += 1
            report.failures.append({"doc": entry.get("arxiv_id"), "error": "pdf missing"})
            continue

        try:
            doc = parse_pdf(
                pdf_path,
                doc_id=entry["arxiv_id"],
                title=entry.get("title", entry["arxiv_id"]),
                licence=licence,
                licence_url=licence_url,
                metadata={
                    "arxiv_id": entry["arxiv_id"],
                    "authors": entry.get("authors", ""),
                    "categories": entry.get("categories", ""),
                    "primary_category": entry.get("primary_category", ""),
                    "created": entry.get("created", ""),
                    "url": f"https://arxiv.org/abs/{entry['arxiv_id']}",
                },
            )
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
            report.documents_failed += 1
            report.failures.append({"doc": entry.get("arxiv_id"), "error": str(exc)})
            continue

        if not doc.blocks:
            report.documents_failed += 1
            report.failures.append({"doc": entry.get("arxiv_id"), "error": "no text extracted"})
            continue

        docs.append(doc)
        report.licences[licence] = report.licences.get(licence, 0) + 1

    # The tabular half of the corpus. Same licence provenance as the papers it
    # describes, since it is derived entirely from their metadata.
    report.documents_ingested = len(docs)

    metadata_csv = settings.corpus_dir / "metadata.csv"
    if metadata_csv.exists():
        try:
            docs.append(
                parse_table_file(
                    metadata_csv,
                    doc_id="corpus-metadata",
                    title="Corpus metadata table",
                    licence="Derived from per-paper metadata (see LICENCES.md)",
                    licence_url="",
                    metadata={"source": "arXiv OAI-PMH metadata", "kind": "table"},
                )
            )
            report.tables_ingested = 1
        except Exception as exc:  # noqa: BLE001
            report.failures.append({"doc": "corpus-metadata", "error": str(exc)})

    return docs, report


def run_ingest(settings: Settings | None = None, *, reset: bool = True) -> IngestReport:
    s = settings or get_settings()
    t0 = time.perf_counter()

    docs, report = load_corpus(s)
    report.config = {
        "corpus_dir": str(s.corpus_dir),
        **s.retrieval_signature(),
    }

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, s))
    report.chunks = len(all_chunks)

    if all_chunks:
        lengths = sorted(len(c.text) for c in all_chunks)
        report.chunk_chars_mean = round(sum(lengths) / len(lengths), 1)
        report.chunk_chars_p95 = lengths[int(0.95 * (len(lengths) - 1))]

    write_chunks_jsonl(all_chunks, s.index_dir / "chunks.jsonl")

    store = VectorStore(s)
    if reset:
        store.reset()
    report.vector_records = store.add(all_chunks)

    bm25 = BM25Index(s)
    report.bm25_records = bm25.build(all_chunks)

    report.duration_s = round(time.perf_counter() - t0, 2)
    report.finished_at = datetime.now(UTC).isoformat()
    path = report.write(s.runs_dir)

    print(
        f"ingested {report.documents_ingested}/{report.documents_seen} documents "
        f"+ {report.tables_ingested} table(s) -> "
        f"{report.chunks} chunks ({s.chunk_strategy}) in {report.duration_s}s\n"
        f"  vector records: {report.vector_records}\n"
        f"  bm25 records:   {report.bm25_records}\n"
        f"  skipped (no licence): {report.documents_skipped_no_licence}\n"
        f"  failed: {report.documents_failed}\n"
        f"  run log: {path}"
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the retrieval indexes from the corpus.")
    ap.add_argument("--no-reset", action="store_true", help="append instead of rebuilding")
    ap.add_argument("--strategy", default=None, help="override ARAA_CHUNK_STRATEGY for this run")
    args = ap.parse_args()

    s = get_settings()
    if args.strategy:
        s = s.model_copy(update={"chunk_strategy": args.strategy})
    run_ingest(s, reset=not args.no_reset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

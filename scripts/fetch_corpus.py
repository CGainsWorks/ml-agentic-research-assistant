#!/usr/bin/env python
"""Fetch a licence-filtered arXiv corpus.

Why OAI-PMH and not the arXiv search API: the search API's Atom feed does not
carry a licence field. The OAI-PMH `arXiv` metadata format does. Since the brief
requires recording the licence of everything ingested, and since roughly two
thirds of arXiv is under a non-redistributable licence, filtering on a field we
can actually read is the only defensible way to build this corpus.

Accepted licences (redistribution + derivative works permitted):

    CC BY 4.0 / 3.0        http://creativecommons.org/licenses/by/...
    CC BY-SA 4.0 / 3.0     http://creativecommons.org/licenses/by-sa/...
    CC0 1.0                http://creativecommons.org/publicdomain/zero/1.0/

Explicitly rejected: the arXiv non-exclusive distribution licence (does not
permit redistribution), and every NC or ND variant.

Outputs, all under data/corpus/:

    pdf/<id>.pdf     the papers
    metadata.csv     one row per paper - this is the tabular half of the corpus
    LICENCES.md      per-document licence record, human readable
    manifest.json    machine-readable manifest incl. the fetch parameters

Nothing here is committed to git; see .gitignore.

Usage:
    python scripts/fetch_corpus.py                 # 200 docs, default window
    python scripts/fetch_corpus.py --limit 60      # smaller, for a quick start
    python scripts/fetch_corpus.py --set cs --from 2024-01-01 --until 2024-03-01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"
PDF_BASE = "https://arxiv.org/pdf"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}

USER_AGENT = (
    "araa-corpus-fetcher/0.1 (https://github.com/CGainsWorks/ml-agentic-research-assistant)"
)

ACCEPTED_LICENCE_PREFIXES = (
    "http://creativecommons.org/licenses/by/4.0",
    "http://creativecommons.org/licenses/by/3.0",
    "http://creativecommons.org/licenses/by-sa/4.0",
    "http://creativecommons.org/licenses/by-sa/3.0",
    "http://creativecommons.org/publicdomain/zero/1.0",
)

LICENCE_NAMES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC BY 3.0",
    "http://creativecommons.org/licenses/by-sa/4.0/": "CC BY-SA 4.0",
    "http://creativecommons.org/licenses/by-sa/3.0/": "CC BY-SA 3.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
}

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: str
    categories: str
    primary_category: str
    created: str
    updated: str
    licence: str
    licence_url: str
    abstract: str
    pdf_path: str = ""


def _http_get(url: str, *, timeout: float = 60.0, retries: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # arXiv throttles with 503 + Retry-After. Honour it rather than
            # hammering; being rude here gets the whole IP blocked.
            if exc.code == 503:
                wait = float(exc.headers.get("Retry-After", delay))
                print(f"    503, waiting {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if exc.code in (429,) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {url}")


def _text(node: ET.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def list_records(oai_set: str, date_from: str, date_until: str, want: int) -> list[Paper]:
    """Page through OAI-PMH, keeping only acceptably-licensed records."""
    kept: list[Paper] = []
    seen: set[str] = set()
    params = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": oai_set,
        "from": date_from,
        "until": date_until,
    }
    url = f"{OAI_ENDPOINT}?{urllib.parse.urlencode(params)}"
    page = 0

    while url and len(kept) < want:
        page += 1
        print(f"[oai] page {page} ({len(kept)}/{want} kept)", file=sys.stderr)
        root = ET.fromstring(_http_get(url))

        error = root.find("oai:error", NS)
        if error is not None:
            raise RuntimeError(f"OAI error: {error.get('code')}: {_text(error)}")

        for record in root.iterfind(".//oai:record", NS):
            meta = record.find(".//ax:arXiv", NS)
            if meta is None:
                continue
            licence_url = _text(meta.find("ax:license", NS))
            if not licence_url.startswith(ACCEPTED_LICENCE_PREFIXES):
                continue

            arxiv_id = _text(meta.find("ax:id", NS))
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)

            authors = "; ".join(
                " ".join(
                    filter(
                        None,
                        [
                            _text(a.find("ax:forenames", NS)),
                            _text(a.find("ax:keyname", NS)),
                        ],
                    )
                )
                for a in meta.iterfind(".//ax:author", NS)
            )
            categories = _text(meta.find("ax:categories", NS))

            kept.append(
                Paper(
                    arxiv_id=arxiv_id,
                    title=" ".join(_text(meta.find("ax:title", NS)).split()),
                    authors=authors,
                    categories=categories,
                    primary_category=categories.split()[0] if categories else "",
                    created=_text(meta.find("ax:created", NS)),
                    updated=_text(meta.find("ax:updated", NS)),
                    licence=LICENCE_NAMES.get(licence_url, licence_url),
                    licence_url=licence_url,
                    abstract=" ".join(_text(meta.find("ax:abstract", NS)).split()),
                )
            )
            if len(kept) >= want:
                break

        token_node = root.find(".//oai:resumptionToken", NS)
        token = _text(token_node)
        if token and len(kept) < want:
            url = f"{OAI_ENDPOINT}?{urllib.parse.urlencode({'verb': 'ListRecords', 'resumptionToken': token})}"
            time.sleep(3.0)  # arXiv asks for >=3s between OAI requests
        else:
            url = ""

    return kept


def download_pdfs(papers: list[Paper], pdf_dir: Path, delay: float) -> list[Paper]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ok: list[Paper] = []
    for i, paper in enumerate(papers, 1):
        dest = pdf_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        # Stored relative to the corpus directory, not the repo root: the corpus
        # commonly lives on a Docker volume outside the repo, and an absolute
        # path would not survive the trip between host and container.
        rel = f"pdf/{dest.name}"
        if dest.exists() and dest.stat().st_size > 10_000:
            paper.pdf_path = rel
            ok.append(paper)
            continue
        print(f"[pdf] {i}/{len(papers)} {paper.arxiv_id}", file=sys.stderr)
        try:
            data = _http_get(f"{PDF_BASE}/{paper.arxiv_id}", timeout=120.0)
        except Exception as exc:  # noqa: BLE001 - a missing PDF is not fatal
            print(f"    skipped ({exc})", file=sys.stderr)
            continue
        if not data.startswith(b"%PDF"):
            print("    skipped (not a PDF; probably a withdrawal notice)", file=sys.stderr)
            continue
        dest.write_bytes(data)
        paper.pdf_path = rel
        ok.append(paper)
        time.sleep(delay)
    return ok


def write_outputs(papers: list[Paper], corpus_dir: Path, params: dict) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = corpus_dir / "metadata.csv"
    fields = [f for f in asdict(papers[0])] if papers else []
    with meta_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            writer.writerow(asdict(p))

    by_licence: dict[str, list[Paper]] = {}
    for p in papers:
        by_licence.setdefault(p.licence, []).append(p)

    lines = [
        "# Corpus licences",
        "",
        "Generated by `scripts/fetch_corpus.py`. Every document in the index appears here.",
        "",
        f"- Source: arXiv OAI-PMH (`{OAI_ENDPOINT}`), set `{params['set']}`",
        f"- Window: {params['from']} to {params['until']}",
        f"- Documents: {len(papers)}",
        "",
        "Only licences permitting redistribution and derivative works were accepted.",
        "The arXiv non-exclusive distribution licence and all NC/ND variants were",
        "rejected at fetch time, not filtered later.",
        "",
        "## Summary",
        "",
        "| Licence | Documents |",
        "| --- | ---: |",
    ]
    for lic, group in sorted(by_licence.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {lic} | {len(group)} |")
    lines += ["", "## Per document", "", "| arXiv ID | Licence | Title |", "| --- | --- | --- |"]
    for p in sorted(papers, key=lambda p: p.arxiv_id):
        safe_title = p.title.replace("|", "\\|")
        lines.append(
            f"| [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id}) | {p.licence} | {safe_title} |"
        )
    (corpus_dir / "LICENCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fetch_params": params,
                "document_count": len(papers),
                "licence_counts": {k: len(v) for k, v in by_licence.items()},
                "documents": [asdict(p) for p in papers],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {len(papers)} documents to {corpus_dir}", file=sys.stderr)
    for lic, group in sorted(by_licence.items(), key=lambda kv: -len(kv[1])):
        print(f"  {lic}: {len(group)}", file=sys.stderr)


def main() -> int:
    default_until = date(2024, 6, 1)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=200, help="target document count")
    ap.add_argument("--set", dest="oai_set", default="cs", help="arXiv OAI set (e.g. cs, stat)")
    ap.add_argument("--from", dest="date_from", default=str(default_until - timedelta(days=45)))
    ap.add_argument("--until", dest="date_until", default=str(default_until))
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between PDF downloads")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "corpus")
    ap.add_argument("--metadata-only", action="store_true", help="skip PDF download")
    args = ap.parse_args()

    params = {
        "set": args.oai_set,
        "from": args.date_from,
        "until": args.date_until,
        "limit": args.limit,
        "accepted_licences": list(ACCEPTED_LICENCE_PREFIXES),
    }

    papers = list_records(args.oai_set, args.date_from, args.date_until, args.limit)
    if not papers:
        print(
            "No acceptably-licensed records in that window. Widen --from/--until.", file=sys.stderr
        )
        return 1

    if not args.metadata_only:
        papers = download_pdfs(papers, args.out / "pdf", args.delay)
        if not papers:
            print("No PDFs downloaded.", file=sys.stderr)
            return 1

    write_outputs(papers, args.out, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

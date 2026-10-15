"""Build the question sets the evals run against.

Two sets, built for different reasons.

`retrieval_queries` — deterministic, no LLM, no cost, byte-identical on every
machine given the same index. A query is a sentence lifted from a paper's own
abstract; the gold set is that paper's body chunks. This is the set the
reranker-lift number is measured on, and it is deterministic precisely so that
"the reranker helped" cannot be an artefact of an LLM having a good day.

Its known weakness, stated here and in the README: finding a paper's body from
its own abstract is easier than answering a real research question, so the
absolute recall numbers are optimistic. It is still a fair *comparison* between
retrieval configurations, because every configuration faces the same queries.

`research_questions` — LLM-generated, chunk-anchored. Each question is written
to be answerable from one specific chunk, and that chunk is recorded as the
gold provenance. This is the harder set, and it is what groundedness is scored
on: it lets the judge ask "does the cited chunk support this claim" against a
known-correct source rather than against a vibe.

Both sets are generated artefacts and are gitignored. Rebuild with:

    python -m evals.questions.build_question_set              # both
    python -m evals.questions.build_question_set --only retrieval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import defaultdict

from araa.config import get_settings
from araa.ingestion.chunking import split_sentences
from araa.ingestion.documents import Chunk
from araa.llm import build_llm
from araa.retrieval.hybrid import ChunkStore
from evals.harness import now_iso, require_index, save_question_set

MIN_QUERY_CHARS = 70
MAX_QUERY_CHARS = 320

# Abstract sentences that are pure boilerplate make useless queries.
_BOILERPLATE = re.compile(
    r"^\s*(our code|code (is|will be) available|we (also )?release|in this paper,? we (present|propose|introduce)\b.{0,20}$)",
    re.IGNORECASE,
)


def _usable_query(sentence: str) -> bool:
    s = sentence.strip()
    if not (MIN_QUERY_CHARS <= len(s) <= MAX_QUERY_CHARS):
        return False
    if _BOILERPLATE.match(s):
        return False
    # Needs enough content words to be a query rather than a fragment.
    return len(s.split()) >= 10


# ------------------------------------------------------- deterministic set
def load_manifest_abstracts(settings) -> dict[str, str]:
    """Abstracts as arXiv published them, from the OAI-PMH manifest.

    Deliberately not the PDF-parsed abstract. The manifest text is complete,
    clean, and present for every document, whereas PDF extraction of the
    abstract depends on the paper's LaTeX template. Using the manifest means the
    query set has full corpus coverage and does not silently shrink whenever the
    parser meets an unfamiliar layout — which is exactly what it did the first
    time this ran: 3 queries from 12 papers.
    """
    manifest = settings.corpus_dir / "manifest.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        d["arxiv_id"]: d.get("abstract", "") for d in data.get("documents", []) if d.get("abstract")
    }


def build_retrieval_queries(
    chunks: list[Chunk],
    per_doc: int,
    seed: int,
    abstracts: dict[str, str] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    abstracts = abstracts or {}

    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)

    questions: list[dict] = []
    for doc_id in sorted(by_doc):
        doc_chunks = by_doc[doc_id]
        if doc_id == "corpus-metadata":
            continue  # the metadata table has no abstract to query from

        abstract_text = abstracts.get(doc_id) or " ".join(
            c.text for c in doc_chunks if c.metadata.get("block_type") == "abstract"
        )
        body = [c for c in doc_chunks if c.metadata.get("block_type") in ("body", "table")]
        # The query is lifted from the abstract, so the abstract chunk itself is
        # a near-duplicate of the query and will sit at rank 1 for every
        # configuration. Scoring it would measure nothing and mask the
        # differences the eval exists to expose, so it is excluded from the
        # retrieved list at scoring time — the standard "drop the query's own
        # source document" correction.
        abstract_chunk_ids = [
            c.chunk_id for c in doc_chunks if c.metadata.get("block_type") == "abstract"
        ]
        if not abstract_text or len(body) < 3:
            continue

        candidates = [s for s in split_sentences(abstract_text) if _usable_query(s)]
        if not candidates:
            continue
        rng.shuffle(candidates)

        for i, sentence in enumerate(candidates[:per_doc]):
            questions.append(
                {
                    "id": f"det-{doc_id}-{i}",
                    "question": sentence.strip(),
                    "source": "abstract_sentence",
                    "gold_doc_ids": [doc_id],
                    # Chunk-level gold is every body chunk of the source paper.
                    # Deliberately generous: an abstract sentence is not
                    # traceable to one specific paragraph, so scoring at chunk
                    # level with a single gold chunk would measure luck.
                    "gold_chunk_ids": [c.chunk_id for c in body],
                    "exclude_chunk_ids": abstract_chunk_ids,
                    "title": doc_chunks[0].metadata.get("title"),
                }
            )
    return questions


# ------------------------------------------------------------- LLM-built set
QUESTION_PROMPT = """You are building an evaluation set for a research assistant.

Below is an excerpt from a research paper. Write ONE specific question that:
- can be answered using only this excerpt,
- names concrete entities from the excerpt (a method, dataset, metric or number),
- would NOT be answerable from general knowledge alone,
- is a natural thing a researcher would ask, not a quiz question about the text.

Do not mention "the excerpt", "this passage", "the paper", or "the authors".
Reply with the question only. No preamble, no quotes, no numbering.

Paper: {title}
Section: {section}

Excerpt:
\"\"\"
{text}
\"\"\"
"""


async def build_research_questions(
    chunks: list[Chunk], count: int, seed: int, min_chars: int
) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    notes: list[str] = []

    pool = [
        c
        for c in chunks
        if c.doc_id != "corpus-metadata"
        and c.metadata.get("block_type") in ("body", "table")
        and len(c.text) >= min_chars
    ]
    if not pool:
        return [], ["no chunks long enough to anchor a question"]

    # At most one question per document, so the set is not dominated by
    # whichever paper happens to chunk into the most paragraphs.
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in pool:
        by_doc[c.doc_id].append(c)
    picked: list[Chunk] = []
    for doc_id in sorted(by_doc):
        picked.append(rng.choice(by_doc[doc_id]))
    rng.shuffle(picked)
    picked = picked[:count]

    llm = build_llm()
    questions: list[dict] = []
    try:
        for i, chunk in enumerate(picked):
            prompt = QUESTION_PROMPT.format(
                title=chunk.metadata.get("title", chunk.doc_id),
                section=chunk.metadata.get("section") or "unspecified",
                text=chunk.text[:2500],
            )
            try:
                gen = await llm.generate(prompt, max_tokens=120)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"generation failed for {chunk.chunk_id}: {exc}")
                continue

            question = _clean_question(gen.text)
            if not question:
                notes.append(f"unusable generation for {chunk.chunk_id}")
                continue

            questions.append(
                {
                    "id": f"gen-{i:03d}",
                    "question": question,
                    "source": "llm_from_chunk",
                    "gold_doc_ids": [chunk.doc_id],
                    "gold_chunk_ids": [chunk.chunk_id],
                    "anchor_chunk_id": chunk.chunk_id,
                    "anchor_text": chunk.text,
                    "title": chunk.metadata.get("title"),
                    "generator_model": gen.model,
                }
            )
            print(f"  [{len(questions)}/{len(picked)}] {question[:90]}", file=sys.stderr)
    finally:
        await llm.aclose()

    return questions, notes


def _clean_question(raw: str) -> str:
    text = raw.strip()
    # Strip a leading "Question:" label, surrounding quotes, list numbering.
    text = re.sub(r"^\s*(question\s*:?\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[\d.)-]+\s*", "", text)
    text = text.strip().strip('"').strip("'").strip()
    # Some small models happily emit a paragraph. Take the first question in it.
    if "\n" in text:
        first = next((ln for ln in text.splitlines() if ln.strip().endswith("?")), "")
        text = (first or text.splitlines()[0]).strip()
    if len(text) < 20 or not text.endswith("?"):
        return ""
    return text


async def _amain() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["retrieval", "research"], default=None)
    ap.add_argument("--per-doc", type=int, default=1, help="deterministic queries per document")
    ap.add_argument("--count", type=int, default=40, help="LLM-generated questions")
    ap.add_argument("--min-chars", type=int, default=500, help="minimum anchor chunk length")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    s = get_settings()
    require_index(s)
    seed = args.seed if args.seed is not None else s.seed

    store = ChunkStore(s)
    chunks = store.all()
    print(f"loaded {len(chunks)} chunks", file=sys.stderr)

    if args.only in (None, "retrieval"):
        abstracts = load_manifest_abstracts(s)
        det = build_retrieval_queries(chunks, args.per_doc, seed, abstracts)
        if not det:
            raise SystemExit(
                "No deterministic queries could be built. Every document needs an "
                "abstract in the corpus manifest and at least 3 body chunks in the index."
            )
        path = save_question_set(
            "retrieval_queries",
            det,
            {
                "built_at": now_iso(),
                "kind": "deterministic",
                "method": "abstract sentence (from the OAI-PMH manifest) -> that paper's body chunks",
                "abstract_source": "manifest" if abstracts else "pdf_parse_fallback",
                "documents_with_abstracts": len(abstracts),
                "per_doc": args.per_doc,
                "seed": seed,
                "chunks_in_index": len(chunks),
                "chunk_strategy": s.chunk_strategy,
            },
        )
        print(f"wrote {len(det)} deterministic queries -> {path}", file=sys.stderr)

    if args.only in (None, "research"):
        print(
            f"generating {args.count} research questions with {s.active_model_name}",
            file=sys.stderr,
        )
        gen, notes = await build_research_questions(chunks, args.count, seed, args.min_chars)
        path = save_question_set(
            "research_questions",
            gen,
            {
                "built_at": now_iso(),
                "kind": "llm_generated",
                "method": "one question per chunk, at most one chunk per document",
                "generator": s.generation_signature(),
                "requested": args.count,
                "produced": len(gen),
                "seed": seed,
                "notes": notes,
            },
        )
        print(f"wrote {len(gen)} research questions -> {path}", file=sys.stderr)
        if notes:
            print(f"  ({len(notes)} generation problems recorded in the metadata)", file=sys.stderr)

    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())

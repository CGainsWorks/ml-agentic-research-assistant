# Corpus, licensing and chunking

## What is ingested

arXiv preprints (computer science), fetched via the **OAI-PMH** interface at
`https://oaipmh.arxiv.org/oai`, plus a derived metadata table.

## Why OAI-PMH and not the arXiv search API

The arXiv search API's Atom feed **does not carry a licence field.** The OAI-PMH
`arXiv` metadata format does. Since roughly two thirds of arXiv is under a
licence that does not permit redistribution, and the brief requires recording
the licence of everything ingested, filtering on a field we can actually read is
the only defensible way to build this corpus. A sample from the `cs` set:

| Licence | Share of a 277-record sample |
| --- | ---: |
| arXiv non-exclusive distribution | 48% — **rejected** |
| CC BY 4.0 | 33% — accepted |
| CC BY-NC-ND 4.0 | 8% — **rejected** |
| CC BY-NC-SA 4.0 | 8% — **rejected** |
| CC BY-SA 4.0 | 2% — accepted |
| CC0 1.0 | 1% — accepted |

## Accepted licences

Redistribution **and** derivative works must both be permitted:

- `CC BY 4.0` / `CC BY 3.0`
- `CC BY-SA 4.0` / `CC BY-SA 3.0`
- `CC0 1.0`

Rejected at fetch time, not filtered later:

- **arXiv non-exclusive distribution licence** — does not permit redistribution.
- **Every NC variant** — non-commercial only. The brief warns about exactly
  this, and a portfolio project that quietly ingests NC data is the failure mode
  it is warning about.
- **Every ND variant** — no derivative works, and chunking a paper into an
  embedded index is a derivative work.

## What is recorded, and where

| Artefact | Contents |
| --- | --- |
| `data/corpus/LICENCES.md` | Every document, its licence, and a link. Human-readable. |
| `data/corpus/manifest.json` | The same, machine-readable, plus the exact fetch parameters. |
| `data/corpus/metadata.csv` | One row per paper. This is also the tabular half of the corpus. |
| Every chunk's metadata | `licence` and `licence_url` travel with the chunk into the index and out into the API response. |

**A document with no recorded licence is never indexed.** Not warned about —
skipped, and counted as such in the ingestion run log. The licence is on every
chunk rather than only in a side file so that an answer citing a passage can
also state the terms that passage is available under.

Nothing under `data/` is committed. `scripts/fetch_corpus.py` regenerates it.

## Rate limiting

arXiv asks for at least 3 seconds between requests and returns `503` with a
`Retry-After` header when it wants you to slow down. The fetcher honours both,
sends a descriptive `User-Agent`, and defaults to a 3-second delay between PDF
downloads. 200 papers therefore takes about ten minutes. Being rude here gets
the whole IP blocked, which would break the project for everyone who clones it.

## Model licences

| Model | Role | Licence |
| --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | Embeddings (33M params) | MIT |
| `BAAI/bge-reranker-base` | Cross-encoder reranking (278M) | MIT |
| `llama3.1:8b` | Generation (default) | Llama 3.1 Community Licence |
| Any OpenAI-compatible model | Generation (optional, `openai_compat`) | Whatever that model's licence or API terms say |

Both embedding models are baked into the Docker image so the eval numbers are
pinned to specific weights rather than to whatever a tag resolves to on the day.

The Llama licence permits commercial use below 700M monthly active users and
requires attribution. Swap it for an Apache-2.0 model
(`ARAA_OLLAMA_MODEL=mistral:7b-instruct`, `qwen2.5:7b-instruct`) if that matters
for your use — nothing in the code depends on the model choice.

---

# Chunking

Four strategies behind one interface, so `make eval-chunking` can hold
everything else fixed and vary only this. "We chose recursive chunking" is an
opinion; "recursive beat fixed by N points on this question set at M% of the
context cost" is a result.

| Strategy | What it does | Where it should win |
| --- | --- | --- |
| `fixed` | Character windows, no overlap. | Nowhere. It is the baseline that makes the others' numbers mean something. |
| `recursive` | Character windows that prefer paragraph → sentence → word boundaries, with overlap. **Default.** | General prose. Overlap keeps a claim split across a boundary retrievable from either side. |
| `sentence_window` | Embeds one sentence, returns it with N sentences of context either side. | Precise factual lookup — the embedding is not diluted by surrounding text, but the model still reads a paragraph. |
| `section_aware` | Respects the document's own section structure; splits only when a section exceeds the budget. | Questions scoped to a part of a paper ("what did they do in the ablation"). |

**Tables are never split by any strategy.** Half a markdown table retrieves
badly and reads worse.

## What the comparison measures

Retrieval quality *and* the cost each strategy imposes: chunk count, mean chunk
length, and the context tokens a query actually consumes. That last column is
the point. A strategy that wins recall by emitting three times as many chunks
and stuffing 4× the characters into the context has not won — it has moved the
cost somewhere the recall number cannot see.

## What it actually found

Measured on the 12-paper corpus with 36 deterministic queries
(`make eval-chunking`, full table in [results.md](results.md)):

| Strategy | Chunks | **doc hit@3** | chunk nDCG@10 | Context tokens |
| --- | ---: | ---: | ---: | ---: |
| `fixed` | 2176 | **91.7%** | 0.689 | 879 |
| `recursive` | 2213 | **91.7%** | 0.639 | **859** |
| `sentence_window` | 5020 | **91.7%** | 0.348 | 1319 |
| `section_aware` | **708** | **91.7%** | 0.689 | 1643 |

**All four tie at 91.7% on the metric that matches the gold granularity.** On
this question set, chunking strategy makes no measurable difference to whether
the right *document* is retrieved.

The chunk-level columns look like they disagree, and they are a trap. Chunk-level
nDCG@10 is computed over a document-level gold set of ~100 chunks, so it is
dominated by how many chunks a strategy emits: `sentence_window` scores 0.348
purely because it produces 5,020 chunks and the top-5 is therefore spread
thinner — its `chunk_hit@10` recovers to 91.7%, and its `doc_hit@3` was never
different. Declaring a winner on nDCG here would be declaring a winner on chunk
count.

So the choice is a cost decision, and `recursive` stays the default on two
grounds: lowest context cost (859 tokens/query), and overlap, which keeps a
claim split across a chunk boundary retrievable from either side.

**Honest caveat on that second ground:** the overlap benefit is *not* what this
eval measured. Document-level gold cannot see a claim severed at a boundary,
because the neighbouring chunk belongs to the same document and scores as a hit
either way. Overlap is retained on reasoning, not on evidence. Measuring it
needs claim-level gold — see the limitations in the README.

## Provenance

Every chunk carries `doc_id`, `title`, `section`, `page`, `licence` and
`licence_url` through chunking, indexing, retrieval and into the citation. That
is what lets a citation read "*LoRA, section 4, page 6*" rather than
"*chunk 7f3a91*", and it is what the groundedness eval checks a claim against.

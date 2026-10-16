# Architecture

## The shape of the thing

```
                          ┌──────────────────────────────────────────┐
   arXiv OAI-PMH ────────►│ fetch_corpus.py                          │
   (licence-filtered)     │   CC-BY / CC-BY-SA / CC0 only            │
                          │   writes LICENCES.md + manifest.json     │
                          └───────────────┬──────────────────────────┘
                                          │  PDFs + metadata.csv
                          ┌───────────────▼──────────────────────────┐
                          │ ingestion/                               │
                          │   pdf.py       two-column reading order, │
                          │                section headings, tables  │
                          │   tabular.py   rows → retrievable text   │
                          │                + a queryable dataframe   │
                          │   chunking.py  4 swappable strategies    │
                          └───────────────┬──────────────────────────┘
                                          │  chunks (with provenance)
                    ┌─────────────────────┴────────────────────┐
                    ▼                                          ▼
        ┌───────────────────────┐                  ┌───────────────────────┐
        │ Chroma (bge-small)    │                  │ BM25 (rank_bm25)      │
        │ dense, cosine         │                  │ sparse, exact tokens  │
        └───────────┬───────────┘                  └───────────┬───────────┘
                    └──────────────┬───────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ Reciprocal Rank Fusion       │
                    │ then bge-reranker cross-enc  │
                    └──────────────┬───────────────┘
                                   │  top-k passages
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ┌────────────────────┐                  ┌────────────────────────┐
   │ single-shot RAG    │                  │ LangGraph agent        │
   │ 1 LLM call         │  ◄── compared ──►│ plan/tools/critique    │
   │ the baseline       │      on the      │ /revise                │
   └─────────┬──────────┘      same eval   └───────────┬────────────┘
             └────────────────────┬────────────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │ citation parse + validation  │
                    │ output guardrails            │
                    └──────────────┬───────────────┘
                                   ▼
                        FastAPI (JSON + SSE) / CLI
```

## Decisions worth defending

### Why the eval harness was written before the agent

The brief says the evals are the point, and building in that order changes what
gets built. Two things in this repository exist only because a metric was
already watching:

* The **retrieval eval's headline metric was wrong on the first run.** With
  document-level gold (~100 relevant chunks per query), chunk-level nDCG@10
  rewards returning ten chunks from one paper. It scored reranking at **−6.2%**
  while `doc_hit@3` simultaneously went from 91.7% to 97.2%. The nDCG figure was
  not evidence the reranker hurt; it was evidence the metric did not match the
  gold. Headline metrics are now chosen per question set
  (`evals/retrieval/run_retrieval_eval.py:HEADLINE_METRIC`).
* The **guardrails blocked a benign control.** A paper *about* prompt injection
  — "prior work shows instructions embedded in retrieved documents can override
  a model's system prompt" — was refused as an attack. The corpus is
  computer-science papers, so that is not a hypothetical. The fix
  (`_is_descriptive`) discriminates on grammatical mood rather than vocabulary.

Neither would have been noticed from reading output.

Two more surfaced only from running the system end to end, and are worth
recording because they are different *kinds* of failure:

* **A crash the type system could not see.** The first real request through the
  agent raised `ValueError: 'plan' is already being used as a state key`.
  LangGraph forbids a node name that collides with a state field, and the node
  producing `plan` was called `plan`. Every unit test passed and every module
  imported. Nodes are now `planner`/`drafter`, and
  `tests/test_graph.py` asserts no node name collides with any state key — so
  the next such collision fails a test naming the offender rather than a
  production request returning a 500.

* **A silent quality failure nothing would have flagged.** The planner's prompt
  included the tool inventory, so the planner dutifully emitted
  `search_corpus query="measurement methods" top_k=10` as its "sub-queries".
  Those strings went into the retriever *as literal search text*. The request
  succeeded, returned a cited answer with no warnings, and every retrieved
  passage came from a single paper. Nothing was raised, no metric was breached,
  and the only symptom was retrieval quietly performing far worse than it
  should. The planner no longer sees the tool inventory (tool selection is a
  separate node with its own prompt), and `_parse_plan` salvages the search
  intent from any line that still looks like a function call.

The second is the more instructive one. It is exactly the class of bug that a
demo hides and an eval catches, which is the argument this whole repository is
making.

### The result that changed what ships

The single-shot baseline exists to answer one question: does the agent actually
beat it? On this corpus it does not.

| | `single_shot` | `agent` |
| --- | ---: | ---: |
| Citation support (LLM judge) | 62.1% | 26.9% |
| Citation support (lexical floor) | 65.5% | 50.0% |
| Hallucination rate | 57.1% | 80.0% |
| Tokens per query | 1,573 | 13,259 |

Two judges sharing no machinery agree on the direction. The agent grounds worse
at 8.4× the cost, and 76% of that cost is critique and revision — the expensive
part is the part doing the damage.

The visible mechanism in the traces: the reviser is told to fix the critic's
issues, and it does so by **rewriting prose**. A reworded claim drifts from what
its cited passage says, so a revision that satisfies the critic can break the
grounding the critic was protecting.

Two consequences, both already in the code:

* `mode` defaults to `single_shot` in `api/schemas.py` and `cli.py`, with the
  measurement recorded at the default. Shipping the agent as the default because
  it sounds more sophisticated is precisely the failure this project is arguing
  against.
* The fix to try next is constraining the reviser to edit *citations* rather
  than prose — and then re-running `make eval-groundedness`, because the point
  is not to have a theory about it.

An agent that loses to its own baseline is not a failed project. It is the
harness working: that comparison is why the baseline is in the eval at all, and
without it this repository would have shipped a slower, more expensive, less
grounded system while describing it as an improvement.

### The bug only a cold start could find

`scripts/bootstrap.sh` carried its own hardcoded model default. `config.py`
moved to `llama3.1:8b`; the script still said `qwen2.5:7b-instruct`. So a
genuinely cold `docker compose up` downloaded 4.7 GB of one model and then
served an API asking for another — `/health` would have reported
`degraded / model not pulled` forever, on the exact path the README tells people
to run first.

Nothing else could have caught it. Unit tests mock the model. The eval harness
runs against an already-built index. Every container test in this project's
development mounted source over a warm volume. The failure existed only in the
transition from nothing to a running stack, which is the one state no other test
occupies — and it surfaced within 60 seconds of the integration test's first
real run.

The fix is not "correct the string". `bootstrap.sh` now reads the model name out
of `araa.config`, so the two cannot disagree, and
`tests/test_bootstrap_consistency.py` fails if anyone reintroduces a literal
default. Fixing the instance leaves the mechanism; removing the mechanism is the
fix.

### Why Ollama is the default and a hosted model is opt-in

Every quantitative claim in this repository must be reproducible by someone who
clones it. If the default path needed a paid API key, "reproducible" would mean
"reproducible if you spend money", and most readers would take the numbers on
trust. Local open-weights inference plus local embedding and reranking models
means the whole eval suite runs for free.

`ARAA_LLM_BACKEND=openai_compat` switches generation to any OpenAI-compatible
chat completions server through the same interface: a self-hosted vLLM or
llama.cpp server, Ollama's own `/v1` route or a hosted API. Its base URL
defaults to the local Ollama, so the switch itself sends nothing off the
machine. Keeping the default local is also a data-residency choice: the corpus
and every question stay on hardware the operator controls. A hosted endpoint is
billed per token, which is a second reason not to make it the default; its
prices are configuration (`ARAA_OPENAI_INPUT_USD_PER_MTOK`,
`ARAA_OPENAI_OUTPUT_USD_PER_MTOK`) so the cost ledger reports real USD for it.

### Why RRF instead of weighted score fusion

Cosine similarity is bounded in [-1, 1]; BM25 is unbounded and corpus-dependent.
Any fixed weighting between them is a magic number that needs re-tuning whenever
the corpus changes, and nothing in the system would tell you it had gone stale.
RRF uses only rank position, so it needs no calibration. `tests/test_retrieval_fusion.py`
pins this: fusion output is identical whether BM25 scores are ~0.4 or ~4000.

### Why the reranker sits at the end

The cross-encoder reads query and passage jointly, which is why it is more
accurate and why it cannot be run over the corpus. Scoring 20 fused candidates
costs ~8 s on a CPU; scoring 2,213 chunks would cost ~15 minutes. It is the last
stage because that is the only place its cost is affordable, and
`ARAA_USE_RERANKER=false` turns it off for anyone who disagrees with the
trade — the eval reports both.

### Why the agent does not use JSON function-calling

The default backend is an 8B open-weights model. Small models emit malformed
JSON often enough that a JSON tool protocol converts "the agent picked the wrong
tool" into "the agent crashed". The line protocol (`TOOL: name | k=v`) has one
bracket to get wrong, degrades to "no tool call" instead of an exception, and
works unchanged on a hosted model. See `src/araa/tools/base.py`.

### Why the agent streams stages, not tokens

`/ask/stream` token-streams the single-shot path, because there is one
generation and every token is final. The agent streams **stage events** and then
the finished answer, because the critique step rewrites drafts. Showing someone
a sentence and then silently replacing it is worse than a short wait.

### Why guardrails run on both sides

Input scanning only catches attacks it recognises; the rule list will always be
incomplete. Output validation catches attacks that *worked*, which does not
depend on having anticipated the payload. The canary is the clearest example: a
random per-process token in the system prompt that the model is told never to
emit. If it appears in an answer, exfiltration succeeded — zero false positives,
no judge required.

The structural defence matters more than either. Retrieved passages are fenced,
labelled untrusted, and the system prompt states that instructions inside them
are data to be reported rather than commands to be followed. A regex list
degrades badly; an instruction hierarchy degrades gracefully.

### Why blocked requests still return a full response

A refusal carries its guardrail findings, its retrieval trace and its cost. A
blocked request is an audit record — it tells the operator that a document in
the corpus is hostile, which is information worth more than the answer.

## Scheduled ingestion: what runs, and what does not

| | Runs today | Written, never applied |
| --- | --- | --- |
| Trigger | APScheduler in `src/araa/ingestion/schedule.py` | EventBridge → Lambda, `infra/terraform/` |
| How | `docker compose --profile scheduler up` | `terraform apply` (never executed) |
| Verified | Yes — runs, writes digests | `terraform validate` + `fmt` in CI only |

Both call the same `scheduled_ingest()`. That is deliberate: one implementation
with two triggers cannot drift, whereas two implementations of "the nightly job"
always do.

**The Terraform has never been applied to an AWS account.** It is checked for
syntax and formatting in CI and nothing more. It is included because the brief
asks for it and because writing the IAM scoping, the alarm on *absence* of
invocations, and the S3 versioning-for-rollback is where the actual design
thinking lives — but calling it "deployed" would be false.

## Layout

```
src/araa/
  config.py        every tunable: retrieval, chunking, models, schedule
  llm/             provider interface + ollama / openai_compat / fixture, cost ledger
  ingestion/       pdf, tabular, chunking strategies, pipeline, scheduler
  retrieval/       embed, chroma, bm25, RRF fusion, cross-encoder rerank
  rag/             prompts, citation parse/validate, single-shot baseline
  graph/           LangGraph state, nodes, graph assembly
  tools/           tool protocol + the four read-only tools
  guardrails/      injection detection, sanitisation, output validation
  api/             streaming FastAPI
evals/
  harness.py       metrics, artefacts, Wilson intervals
  questions/       deterministic + LLM-generated question sets
  retrieval/       retrieval eval, chunking comparison
  groundedness/    citation-support judging
  adversarial/     attacks.yaml + two-tier suite
  costs/           token and cost attribution
  report.py        condenses the artefacts into docs/results.json, renders docs/results.md
infra/
  terraform/       scheduled ingestion on AWS (NOT APPLIED)
  lambda/          the handler that Terraform would invoke
```

# Status

Last updated: 2026-10-16

## Complete

- Licence-filtered arXiv corpus fetch over OAI-PMH, PDF parsing, four chunking
  strategies, hybrid BM25 plus dense retrieval with RRF and a cross-encoder
  reranker.
- Single-shot RAG with citation grounding, and a LangGraph agent (plan, tools,
  critique, revise). The agent measures worse than the baseline and is opt-in.
- Injection guardrails with a tiered adversarial suite; tier 1 runs in CI.
- Streaming FastAPI service, APScheduler re-ingest worker, per-node cost ledger.
- Eval harness. `docs/results.json` is committed and condensed from the raw
  artefacts by `python -m evals.report --from-runs`, with each artefact's
  SHA-256, run time and configuration. `docs/results.md` is rendered from it,
  CI fails if the two disagree (`python -m evals.report --check`), and
  `tests/test_results_doc.py` fails if a figure quoted in the README differs
  from `docs/results.json`.
- One corpus size everywhere: 12 papers, the corpus the published results were
  measured on (`araa.config`, `scripts/fetch_corpus.py`, `scripts/bootstrap.sh`,
  `.env.example`, Terraform and the Lambda handler). A test pins them together.
- Paid API emulation: CI never calls a model API. Unit tests use the `fixture`
  backend and `httpx.MockTransport`; `tests/test_mock_openai_server.py` drives
  the `openai_compat` backend over a real socket against
  `scripts/mock_openai_server.py`; the e2e workflow runs the compose stack with
  that mock as the generator (`docker-compose.mock-llm.yml`).
- 274 tests (`make test`), ruff lint and format, `terraform validate`, Docker
  build and smoke in CI.

## In progress

- Nothing.

## Not started

- Re-running the evals at corpus scale (hundreds of papers) and with a judge
  model different from the author model.
- Constraining the reviser to edit citations, then re-measuring the agent.
- Upload of a user's own documents and per-tenant collections.

## Limitations

- The published numbers come from eval runs on 2026-08-05 and 2026-08-09
  (llama3.1:8b through Ollama, on CPU). The raw per-question artefacts are not
  committed; `docs/results.json` holds every number the docs quote. Two
  wording-only transformations were applied when it was built and are listed
  in it: projection price points relabelled by tier (matched on identical
  rates) and the self-judging warning re-rendered with the current flag name.
  No number changed.
- The mock generator returns canned answers, so the e2e run proves the stack's
  plumbing (HTTP, streaming, usage accounting, citations pointing at real
  passages), not answer quality. Answer quality is only measured by the evals,
  which need a real model.
- The planner before and after table in the README, the licence-share sample in
  `docs/corpus.md` and the debugging anecdotes in `docs/architecture.md` come
  from hand-read traces and one-off samples, not from committed artefacts.
- The AWS Terraform and Lambda handler are validated in CI but have never been
  applied. The APScheduler worker is the scheduled ingestion that runs.
- Latency figures are from a CPU run and are not benchmarks.

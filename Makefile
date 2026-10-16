.DEFAULT_GOAL := help
SHELL := /bin/bash

PY ?= python

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ the demo
.PHONY: demo
demo: ## Everything, one command (docker compose up)
	docker compose up --build

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean
clean: ## Stop the stack and delete corpus/index volumes
	docker compose down -v

# ------------------------------------------------------------------- local
.PHONY: install
install: ## Install the package + dev extras into the current environment
	$(PY) -m pip install -e ".[dev]"

.PHONY: corpus
corpus: ## Fetch the CC-BY arXiv corpus (records a licence per document)
	$(PY) scripts/fetch_corpus.py

.PHONY: ingest
ingest: ## Parse, chunk, embed and index the corpus
	$(PY) -m araa.ingestion.pipeline

.PHONY: serve
serve: ## Run the streaming API locally
	uvicorn araa.api.app:app --host 0.0.0.0 --port 8000 --reload

.PHONY: ask
ask: ## One-off query: make ask Q="your question"
	$(PY) -m araa.cli ask "$(Q)"

# -------------------------------------------------------------------- evals
# Every number quoted in the README is produced by one of these targets.
.PHONY: eval
eval: eval-retrieval eval-chunking eval-groundedness eval-adversarial eval-cost ## Run every eval

.PHONY: eval-retrieval
eval-retrieval: ## Recall@k / nDCG@k, and the reranker's measured lift over vector-only
	$(PY) -m evals.retrieval.run_retrieval_eval

.PHONY: eval-chunking
eval-chunking: ## Compare the four chunking strategies on the same question set
	$(PY) -m evals.retrieval.run_chunking_comparison

.PHONY: eval-groundedness
eval-groundedness: ## How often a citation actually supports the claim it is attached to
	$(PY) -m evals.groundedness.run_groundedness

.PHONY: eval-adversarial
eval-adversarial: ## Prompt-injection suite; guardrails must pass it
	$(PY) -m evals.adversarial.run_injection_suite

.PHONY: eval-cost
eval-cost: ## Token and USD cost per query, attributed by graph node
	$(PY) -m evals.costs.run_cost_report

.PHONY: report
report: ## Rebuild docs/results.json from data/runs/ and render docs/results.md
	$(PY) -m evals.report --from-runs

.PHONY: report-check
report-check: ## Fail if docs/results.md does not match docs/results.json (CI runs this)
	$(PY) -m evals.report --check

# --------------------------------------------------------------------- CI
.PHONY: test
test: ## Unit tests (no model server needed)
	$(PY) -m pytest -q

.PHONY: e2e
e2e: ## End-to-end integration test against a running stack
	./scripts/e2e_test.sh

.PHONY: e2e-cold
e2e-cold: ## Destroy volumes, docker compose up, then assert the whole stack (real model)
	./scripts/e2e_test.sh --up --fresh

.PHONY: e2e-mock
e2e-mock: ## As e2e-cold, with the mock OpenAI-compatible server as the generator
	COMPOSE_PATH_SEPARATOR=: COMPOSE_FILE=docker-compose.yml:docker-compose.mock-llm.yml ./scripts/e2e_test.sh --up --fresh

.PHONY: lint
lint: ## Lint + format check
	ruff check .
	ruff format --check .

.PHONY: fmt
fmt: ## Autoformat
	ruff format .
	ruff check --fix .

.PHONY: tf-validate
tf-validate: ## Validate the Terraform. Never applied — see docs/architecture.md
	cd infra/terraform && terraform init -backend=false && terraform validate && terraform fmt -check

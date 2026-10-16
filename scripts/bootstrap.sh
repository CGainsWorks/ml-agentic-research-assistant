#!/usr/bin/env bash
# One-shot setup run by the `bootstrap` service in docker-compose.yml.
#
# Idempotent by design: every step checks whether its output already exists, so
# `docker compose up` a second time skips straight to serving.
set -euo pipefail

CORPUS_DIR="${ARAA_CORPUS_DIR:-/data/corpus}"
INDEX_DIR="${ARAA_INDEX_DIR:-/data/index}"
OLLAMA_URL="${ARAA_OLLAMA_BASE_URL:-http://ollama:11434}"
BACKEND="${ARAA_LLM_BACKEND:-ollama}"

# Read the model name out of the package rather than hardcoding it here.
#
# This file used to carry its own default, and it drifted: config.py moved to
# llama3.1:8b while this still said qwen2.5:7b-instruct. The result was a stack
# that pulled one model on boot and then served an API asking for a different
# one — a 4.7 GB download followed by a permanently degraded /health. Asking the
# package makes that class of drift impossible rather than merely fixed.
OLLAMA_MODEL="${ARAA_OLLAMA_MODEL:-}"
if [ -z "$OLLAMA_MODEL" ]; then
  OLLAMA_MODEL=$(python -c 'from araa.config import get_settings; print(get_settings().ollama_model)' 2>/dev/null) || OLLAMA_MODEL=""
fi
if [ -z "$OLLAMA_MODEL" ]; then
  echo "[bootstrap] could not determine the model from araa.config" >&2
  exit 1
fi
# Corpus size comes from the package too (ARAA_CORPUS_TARGET_DOCS overrides
# it), so compose, `make corpus` and the published results agree on one number.
DOCS=$(python -c 'from araa.config import get_settings; print(get_settings().corpus_target_docs)' 2>/dev/null) || DOCS=""
if [ -z "$DOCS" ]; then
  echo "[bootstrap] could not determine the corpus size from araa.config" >&2
  exit 1
fi

log() { printf '\n\033[1m[bootstrap]\033[0m %s\n' "$*"; }

# ------------------------------------------------------------------ 1. model
if [ "$BACKEND" = "ollama" ]; then
  log "waiting for ollama at ${OLLAMA_URL}"
  for _ in $(seq 1 60); do
    if curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then break; fi
    sleep 2
  done

  if curl -fsS "${OLLAMA_URL}/api/tags" | grep -q "\"${OLLAMA_MODEL}\""; then
    log "model ${OLLAMA_MODEL} already present"
  else
    log "pulling ${OLLAMA_MODEL} (first run only; this is the slow part)"
    curl -fsS -X POST "${OLLAMA_URL}/api/pull" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${OLLAMA_MODEL}\"}" \
      | grep -o '"status":"[^"]*"' | uniq | tail -20 || true
  fi
else
  log "backend is '${BACKEND}', skipping ollama model pull"
fi

# ----------------------------------------------------------------- 2. corpus
if [ -f "${CORPUS_DIR}/manifest.json" ]; then
  log "corpus already present at ${CORPUS_DIR}, skipping fetch"
else
  log "fetching ${DOCS} CC-BY licensed arXiv papers (rate-limited, be patient)"
  python /app/scripts/fetch_corpus.py --limit "${DOCS}" --out "${CORPUS_DIR}"
fi

# ------------------------------------------------------------------ 3. index
if [ -f "${INDEX_DIR}/chunks.jsonl" ] && [ -f "${INDEX_DIR}/bm25.json" ]; then
  log "index already built at ${INDEX_DIR}, skipping ingest"
else
  log "parsing, chunking, embedding and indexing"
  python -m araa.ingestion.pipeline
fi

log "ready. API will come up on http://localhost:8000"

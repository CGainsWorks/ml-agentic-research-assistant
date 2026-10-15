#!/usr/bin/env bash
# Bring the stack up, wait for it, then ask it something and show the answer,
# the citations, the retrieval trace and the cost.
set -euo pipefail

cd "$(dirname "$0")/.."

QUESTION=${1:-"What approaches do these papers take to improving retrieval quality, and what do they measure?"}
API=${API:-http://localhost:8000}

echo "==> docker compose up (first run downloads a model and a corpus)"
docker compose up -d --build

echo "==> waiting for the API to become healthy"
for _ in $(seq 1 180); do
  if curl -fsS "${API}/health" >/dev/null 2>&1; then break; fi
  sleep 5
done

if ! curl -fsS "${API}/health" >/dev/null 2>&1; then
  echo "API never came up. Logs:" >&2
  docker compose logs --tail 60 >&2
  exit 1
fi

echo
echo "==> health"
curl -fsS "${API}/health" | python -m json.tool

echo
echo "==> streaming answer to: ${QUESTION}"
curl -fsS -N -X POST "${API}/ask/stream" \
  -H 'Content-Type: application/json' \
  -d "{\"question\": $(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$QUESTION")}"

echo
echo "==> same question, structured JSON (citations, trace, cost)"
curl -fsS -X POST "${API}/ask" \
  -H 'Content-Type: application/json' \
  -d "{\"question\": $(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$QUESTION")}" \
  | python -m json.tool

echo
echo "Stop with: docker compose down   (add -v to delete the corpus and index)"

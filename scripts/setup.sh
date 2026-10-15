#!/usr/bin/env bash
# Local (non-Docker) setup. Prefer `docker compose up` — this is for developing
# on the code itself, where a rebuild per edit is tiresome.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python3}

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11+ required; found: $("$PY" --version)" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing CPU torch (the GPU wheel is 2.5GB and buys nothing here)"
pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

echo "==> installing araa"
pip install --quiet -e ".[dev]"

echo "==> caching embedding + reranker weights"
python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("BAAI/bge-reranker-base")
print("ok")
PY

cat <<'EOF'

Setup complete. Next:

    source .venv/bin/activate
    make corpus     # fetch the CC-BY arXiv corpus (records a licence per doc)
    make ingest     # parse, chunk, embed, index
    make eval       # every number quoted in the README
    make serve      # streaming API on http://localhost:8000

Generation defaults to a local Ollama at http://localhost:11434. Start one with:

    ollama serve && ollama pull qwen2.5:7b-instruct

EOF

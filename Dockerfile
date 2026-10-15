FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models \
    # Chroma 0.5.x still tries to emit telemetry even when it is disabled in
    # code, and fails noisily. Belt and braces.
    ANONYMIZED_TELEMETRY=False \
    CHROMA_TELEMETRY_IMPL=chromadb.telemetry.product.noop \
    # Model inference is already batched; letting torch grab every core makes
    # it slower, not faster, inside a container with a CPU quota.
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch. The GPU wheel is ~2.5GB and buys nothing here: bge-small is
# 33M params and the reranker is 278M, both comfortable on CPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -e .

# Bake the embedding + reranker weights into the image so `docker compose up`
# does not need to reach huggingface.co at runtime, and so the eval numbers are
# pinned to specific weights rather than to whatever "latest" resolves to.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("BAAI/bge-reranker-base")
print("embedding + reranker weights cached")
PY

COPY evals/ ./evals/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
# The Terraform points the Lambda at `infra.lambda.handler.handler` inside this
# image, so the handler has to actually be in it. It is never invoked by the
# compose stack; it is here so the image the Terraform describes is the image
# that exists.
COPY infra/ ./infra/

EXPOSE 8000

CMD ["uvicorn", "araa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

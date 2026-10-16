"""Central configuration.

Every knob the brief names as an input lives here: retrieval config, chunking
parameters, model and temperature settings, and the ingestion schedule. Values
come from environment variables (see `.env.example`) so a run is reproducible
from a recorded env rather than from code edits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

ChunkStrategy = Literal["fixed", "recursive", "sentence_window", "section_aware"]
LLMBackend = Literal["ollama", "openai_compat", "fixture"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARAA_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    # ---------- paths ----------
    data_dir: Path = REPO_ROOT / "data"
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    chroma_dir: Path = REPO_ROOT / "data" / "chroma"
    index_dir: Path = REPO_ROOT / "data" / "index"
    runs_dir: Path = REPO_ROOT / "data" / "runs"

    # ---------- generation ----------
    llm_backend: LLMBackend = "ollama"
    # Ollama (default: open source, no key, no cost).
    # llama3.1:8b rather than a reasoning model: this pipeline makes 4-6 calls
    # per query and most of them are short structured outputs (a plan, a tool
    # line, a verdict). Reasoning models spend a large token budget thinking
    # before emitting three words, which is the wrong trade here. Reasoning
    # models still work - see `ollama_disable_thinking`.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    # qwen3 and friends emit <think>...</think> before the answer, which would
    # otherwise be parsed as part of the answer and wreck citation extraction.
    ollama_disable_thinking: bool = True
    # Any OpenAI-compatible chat completions endpoint (optional): OpenAI, vLLM,
    # llama.cpp server, LM Studio, a gateway or Ollama's own /v1 route. The
    # default points at a local Ollama so switching backend sends nothing off
    # the machine until the base URL is changed on purpose.
    openai_base_url: str = "http://localhost:11434/v1"
    openai_model: str = "llama3.1:8b"
    # The NAME of the environment variable holding the API key, not the key
    # itself, so a key never lands in a settings dump or an eval artefact.
    # Local servers usually need no key; leave the variable unset for those.
    openai_api_key_env: str = "OPENAI_API_KEY"
    # USD per million tokens for the configured model. Zero for self-hosted
    # models, which have no marginal API cost. Set these from the provider's
    # price list to make cost_usd real for a hosted endpoint.
    openai_input_usd_per_mtok: float = 0.0
    openai_output_usd_per_mtok: float = 0.0
    # Deterministic replay of recorded generations. Used by CI so the whole
    # pipeline is exercised without a model server or an API key.
    fixture_path: Path = REPO_ROOT / "evals" / "fixtures" / "generations.jsonl"

    temperature: float = 0.0
    max_output_tokens: int = 900
    # Generous on purpose. If the model does not fit in VRAM, Ollama silently
    # falls back to CPU and a 3k-token prompt goes from seconds to minutes.
    # Measured on this project's dev box with the GPU occupied: 216s for a
    # 3.1k-token prompt. A 180s timeout turns that into a crash rather than a
    # slow answer, which is the wrong failure.
    request_timeout_s: float = 600.0

    # ---------- embeddings + reranking ----------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    # bge models want this prefix on the *query* side only.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = 16

    # ---------- chunking ----------
    chunk_strategy: ChunkStrategy = "recursive"
    chunk_size: int = 900  # characters
    chunk_overlap: int = 150  # characters
    sentence_window_size: int = 3  # sentences either side, for sentence_window

    # ---------- retrieval ----------
    vector_top_k: int = 30
    bm25_top_k: int = 30
    rrf_k: int = 60  # RRF damping constant
    fused_top_k: int = 20  # candidates handed to the reranker
    final_top_k: int = 6  # context chunks handed to the LLM
    use_reranker: bool = True
    # Prompt-evaluation time dominates on CPU, and it scales with context. 1200
    # chars x 6 chunks is ~1.9k tokens of evidence, which fits comfortably and
    # still carries whole paragraphs. Raise it when running on a GPU.
    max_context_chars_per_chunk: int = 1200

    collection_name: str = "araa_chunks"

    # ---------- agent ----------
    max_plan_steps: int = 4
    max_critique_rounds: int = 2
    max_tool_calls: int = 8

    # ---------- guardrails ----------
    guardrails_enabled: bool = True
    guardrail_block_on_detect: bool = True

    # ---------- ingestion schedule ----------
    # APScheduler cron for the re-embedding worker. Default: 03:15 daily.
    ingest_cron: str = "15 3 * * *"
    ingest_enabled: bool = False
    # The 12 papers docs/results.md was measured on (scripts/fetch_corpus.py's
    # default window). Raise it for a bigger corpus; the published numbers
    # then no longer apply.
    corpus_target_docs: int = 12

    # ---------- api ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    seed: int = 1234

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.corpus_dir, self.chroma_dir, self.index_dir, self.runs_dir):
            p.mkdir(parents=True, exist_ok=True)

    def retrieval_signature(self) -> dict:
        """The subset of config that changes retrieval results.

        Recorded in every eval artefact so a number in the README can be traced
        back to the exact configuration that produced it.
        """
        return {
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "sentence_window_size": self.sentence_window_size,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model if self.use_reranker else None,
            "vector_top_k": self.vector_top_k,
            "bm25_top_k": self.bm25_top_k,
            "rrf_k": self.rrf_k,
            "fused_top_k": self.fused_top_k,
            "final_top_k": self.final_top_k,
            "use_reranker": self.use_reranker,
        }

    def generation_signature(self) -> dict:
        return {
            "llm_backend": self.llm_backend,
            "model": self.active_model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    @property
    def active_model_name(self) -> str:
        if self.llm_backend == "openai_compat":
            return self.openai_model
        if self.llm_backend == "ollama":
            return self.ollama_model
        return "fixture"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    """Test hook — drop the cached Settings so env changes take effect."""
    get_settings.cache_clear()


# Per-strategy defaults used by the chunking comparison so each strategy is
# evaluated at a sensible operating point rather than at one shared size.
CHUNKING_GRID: dict[str, dict] = {
    "fixed": {"chunk_size": 900, "chunk_overlap": 0},
    "recursive": {"chunk_size": 900, "chunk_overlap": 150},
    "sentence_window": {"sentence_window_size": 3},
    "section_aware": {"chunk_size": 1400, "chunk_overlap": 100},
}

# Prices in USD per million tokens. Only used for cost accounting/reporting.
# Local models cost nothing to call; the electricity cost is real but not
# something this repo pretends to measure.
#
# These are reference price points for projecting measured token counts onto a
# hosted model, labelled by tier rather than by vendor model ID. The live
# `openai_compat` backend does not read this table: it is priced from
# `openai_input_usd_per_mtok` / `openai_output_usd_per_mtok`.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # label: (input, output)
    "hosted-llm-small": (1.00, 5.00),
    "hosted-llm-large": (3.00, 15.00),
}


def price_for(model: str) -> tuple[float, float]:
    """(input, output) USD per million tokens. Local models are (0.0, 0.0)."""
    return PRICING_USD_PER_MTOK.get(model, (0.0, 0.0))

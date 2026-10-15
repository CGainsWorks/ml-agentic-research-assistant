from __future__ import annotations

import pytest

from araa.config import Settings, price_for
from araa.llm.base import Generation, Usage
from araa.llm.cost import CostLedger


def gen(node_tokens_in: int, out: int, model: str = "hosted-llm-small") -> Generation:
    return Generation(
        text="x",
        usage=Usage(input_tokens=node_tokens_in, output_tokens=out),
        model=model,
        latency_s=0.5,
    )


def test_ledger_totals_and_attribution():
    ledger = CostLedger()
    ledger.record("plan", gen(100, 20))
    ledger.record("draft", gen(2000, 300))
    ledger.record("critique", gen(2200, 80))

    assert ledger.total_input_tokens == 4300
    assert ledger.total_output_tokens == 400
    assert ledger.total_tokens == 4700

    by_node = ledger.by_node()
    assert set(by_node) == {"plan", "draft", "critique"}
    assert by_node["draft"]["input_tokens"] == 2000


def test_cost_uses_the_model_price():
    ledger = CostLedger()
    ledger.record("draft", gen(1_000_000, 1_000_000, "hosted-llm-small"))
    # hosted-llm-small reference price point: $1 in, $5 out per Mtok
    assert ledger.total_cost_usd == pytest.approx(6.0)


def test_local_models_have_no_marginal_api_cost():
    ledger = CostLedger()
    ledger.record("draft", gen(50_000, 5_000, "llama3.1:8b"))
    assert ledger.total_cost_usd == 0.0
    # ...but the tokens are still real and still reported
    assert ledger.total_tokens == 55_000


def test_unknown_model_prices_at_zero_rather_than_guessing():
    assert price_for("some-model-we-have-never-heard-of") == (0.0, 0.0)


def test_ledger_serialises_for_the_api():
    ledger = CostLedger()
    ledger.record("draft", gen(10, 5))
    d = ledger.to_dict()
    assert d["llm_calls"] == 1
    assert d["total_tokens"] == 15
    assert "draft" in d["by_node"]


# ----------------------------------------------------------------- settings
def test_retrieval_signature_captures_what_changes_results(tmp_path):
    s = Settings(data_dir=tmp_path)
    sig = s.retrieval_signature()
    for key in ("chunk_strategy", "chunk_size", "embedding_model", "final_top_k", "use_reranker"):
        assert key in sig


def test_disabling_the_reranker_shows_in_the_signature(tmp_path):
    s = Settings(data_dir=tmp_path, use_reranker=False)
    assert s.retrieval_signature()["reranker_model"] is None


def test_active_model_name_follows_the_backend(tmp_path):
    assert Settings(data_dir=tmp_path, llm_backend="ollama").active_model_name.startswith("llama")
    assert (
        Settings(
            data_dir=tmp_path, llm_backend="openai_compat", openai_model="my-served-model"
        ).active_model_name
        == "my-served-model"
    )
    assert Settings(data_dir=tmp_path, llm_backend="fixture").active_model_name == "fixture"

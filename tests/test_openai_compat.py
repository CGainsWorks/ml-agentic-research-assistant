"""The OpenAI-compatible backend, against a mocked HTTP transport.

No server and no network: `httpx.MockTransport` answers every request, so
these pin the request shape the backend sends and how it reads token usage
back, which is what cost accounting depends on.
"""

from __future__ import annotations

import json

import httpx
import pytest

from araa.config import Settings
from araa.llm import CostLedger, build_llm
from araa.llm.openai_compat import OpenAICompatLLM, resolve_api_key


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "llm_backend": "openai_compat",
        "openai_base_url": "http://llm.test/v1/",
        "openai_model": "served-model",
        "openai_api_key_env": "ARAA_TEST_OPENAI_KEY",
    }
    return Settings(**(base | overrides))


def _completion(text: str, prompt_tokens: int = 120, completion_tokens: int = 30) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class Recorder:
    def __init__(self, handler):
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def test_factory_builds_the_openai_compat_backend(tmp_path):
    llm = build_llm(_settings(tmp_path))
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model_name == "served-model"


async def test_generate_sends_a_chat_completion_and_reads_usage(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAA_TEST_OPENAI_KEY", raising=False)
    rec = Recorder(lambda r: httpx.Response(200, json=_completion("An answer [C1].")))
    llm = OpenAICompatLLM(_settings(tmp_path), transport=httpx.MockTransport(rec))
    try:
        gen = await llm.generate("the question", system="the rules", max_tokens=50)
    finally:
        await llm.aclose()

    (req,) = rec.requests
    assert req.method == "POST"
    assert str(req.url) == "http://llm.test/v1/chat/completions"
    body = json.loads(req.content)
    assert body["model"] == "served-model"
    assert body["messages"] == [
        {"role": "system", "content": "the rules"},
        {"role": "user", "content": "the question"},
    ]
    assert body["max_tokens"] == 50
    assert body["stream"] is False
    # No key configured: no Authorization header, which is what local servers want.
    assert "authorization" not in req.headers

    assert gen.text == "An answer [C1]."
    assert gen.usage.input_tokens == 120
    assert gen.usage.output_tokens == 30
    assert gen.model == "served-model"
    assert gen.raw["usage_reported"] is True


async def test_api_key_comes_from_the_named_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAA_TEST_OPENAI_KEY", "test-key-value")
    rec = Recorder(lambda r: httpx.Response(200, json=_completion("ok")))
    llm = OpenAICompatLLM(_settings(tmp_path), transport=httpx.MockTransport(rec))
    try:
        await llm.generate("q")
    finally:
        await llm.aclose()
    assert rec.requests[0].headers["authorization"] == "Bearer test-key-value"


def test_the_key_is_never_a_settings_field():
    """Only the variable's name is configuration, so a settings dump or an eval
    artefact cannot carry the key."""
    assert "openai_api_key" not in Settings.model_fields
    assert "openai_api_key_env" in Settings.model_fields


def test_unset_key_env_var_resolves_to_none(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("ARAA_TEST_OPENAI_KEY", raising=False)
    assert resolve_api_key("ARAA_TEST_OPENAI_KEY") is None
    assert resolve_api_key("") is None


async def test_missing_usage_is_reported_as_zero_and_flagged(tmp_path):
    payload = {"choices": [{"message": {"content": "no usage block"}}]}
    llm = OpenAICompatLLM(
        _settings(tmp_path),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    try:
        gen = await llm.generate("q")
    finally:
        await llm.aclose()
    assert gen.usage.total_tokens == 0
    assert gen.raw["usage_reported"] is False


async def test_thinking_blocks_are_stripped(tmp_path):
    payload = _completion("<think>deliberation</think>The answer [C2].")
    llm = OpenAICompatLLM(
        _settings(tmp_path),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    try:
        gen = await llm.generate("q")
    finally:
        await llm.aclose()
    assert gen.text == "The answer [C2]."


async def test_configured_price_flows_into_the_cost_ledger(tmp_path):
    settings = _settings(tmp_path, openai_input_usd_per_mtok=2.0, openai_output_usd_per_mtok=8.0)
    payload = _completion("x", prompt_tokens=1_000_000, completion_tokens=500_000)
    llm = OpenAICompatLLM(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    ledger = CostLedger()
    try:
        ledger.record("draft", await llm.generate("q"))
    finally:
        await llm.aclose()
    # $2 x 1M input + $8 x 0.5M output
    assert ledger.total_cost_usd == pytest.approx(6.0)
    assert ledger.to_dict()["by_node"]["draft"]["cost_usd"] == pytest.approx(6.0)


async def test_unpriced_model_costs_nothing_but_still_counts_tokens(tmp_path):
    payload = _completion("x", prompt_tokens=4000, completion_tokens=300)
    llm = OpenAICompatLLM(
        _settings(tmp_path),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    ledger = CostLedger()
    try:
        ledger.record("draft", await llm.generate("q"))
    finally:
        await llm.aclose()
    assert ledger.total_cost_usd == 0.0
    assert ledger.total_tokens == 4300


async def test_stream_yields_deltas_and_records_final_usage(tmp_path):
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world [C1]."}}]},
        {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 7}},
    ]
    sse = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    rec = Recorder(
        lambda r: httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    settings = _settings(tmp_path, openai_input_usd_per_mtok=1.0, openai_output_usd_per_mtok=1.0)
    llm = OpenAICompatLLM(settings, transport=httpx.MockTransport(rec))
    try:
        pieces = [p async for p in llm.stream("q")]
    finally:
        await llm.aclose()

    assert "".join(pieces) == "Hello world [C1]."
    body = json.loads(rec.requests[0].content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert llm.last_stream_usage.input_tokens == 42
    assert llm.last_stream_usage.output_tokens == 7

    ledger = CostLedger()
    ledger.record_usage(
        "answer", llm.model_name, llm.last_stream_usage, price_per_mtok=llm.price_per_mtok
    )
    assert ledger.total_cost_usd == pytest.approx(49 / 1_000_000)


async def test_health_and_model_listing_use_the_models_route(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "served-model"}, {"id": "other"}]})

    llm = OpenAICompatLLM(_settings(tmp_path), transport=httpx.MockTransport(handler))
    try:
        assert await llm.health() is True
        assert await llm.list_models() == ["served-model", "other"]
    finally:
        await llm.aclose()


async def test_unreachable_server_reports_unhealthy(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm = OpenAICompatLLM(_settings(tmp_path), transport=httpx.MockTransport(handler))
    try:
        assert await llm.health() is False
    finally:
        await llm.aclose()

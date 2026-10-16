"""The openai_compat backend against the mock server, over real HTTP.

`tests/test_openai_compat.py` pins the request shape with an in-process
`httpx.MockTransport`. These start `scripts/mock_openai_server.py` on a
loopback port and talk to it through a real socket, so the server the e2e
workflow uses is itself tested, and the backend is exercised end to end:
plain and streamed completions, usage accounting and the /models health check.
"""

from __future__ import annotations

import importlib.util
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from araa.config import Settings
from araa.llm.openai_compat import OpenAICompatLLM

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mock_openai_server.py"


def _load_mock():
    if not SCRIPT.exists():
        pytest.skip("scripts/ not present in this build context")
    spec = importlib.util.spec_from_file_location("mock_openai_server", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_url():
    mock = _load_mock()
    server = ThreadingHTTPServer(("127.0.0.1", 0), mock.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _llm(tmp_path, url: str) -> OpenAICompatLLM:
    return OpenAICompatLLM(
        Settings(
            data_dir=tmp_path,
            llm_backend="openai_compat",
            openai_base_url=url,
            openai_model="mock-llm",
            openai_api_key_env="ARAA_TEST_UNSET_KEY",
        )
    )


CONTEXT = "Question: what?\n\n[C1] (paper a)\ntext one\n\n[C2] (paper b)\ntext two"


async def test_generate_cites_only_supplied_passages_and_reports_usage(tmp_path, mock_url):
    llm = _llm(tmp_path, mock_url)
    try:
        gen = await llm.generate(CONTEXT, system="answer with citations")
    finally:
        await llm.aclose()
    assert "[C1]" in gen.text and "[C2]" in gen.text
    assert "[C3]" not in gen.text
    assert gen.usage.input_tokens > 0 and gen.usage.output_tokens > 0
    assert gen.raw["usage_reported"] is True


async def test_stream_yields_deltas_and_final_usage(tmp_path, mock_url):
    llm = _llm(tmp_path, mock_url)
    try:
        pieces = [p async for p in llm.stream(CONTEXT, system="answer")]
    finally:
        await llm.aclose()
    assert len(pieces) > 1
    assert "[C1]" in "".join(pieces)
    assert llm.last_stream_usage.output_tokens > 0


async def test_health_and_model_listing(tmp_path, mock_url):
    llm = _llm(tmp_path, mock_url)
    try:
        assert await llm.health() is True
        assert await llm.list_models() == ["mock-llm"]
    finally:
        await llm.aclose()


def test_each_pipeline_step_gets_the_format_it_parses():
    from araa.graph.nodes import _parse_critique, _parse_plan
    from araa.rag.prompts import CRITIC_SYSTEM, PLANNER_SYSTEM, TOOL_SELECTOR_SYSTEM

    mock = _load_mock()
    say = lambda system, user="q": mock.reply_for(  # noqa: E731
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    assert say(TOOL_SELECTOR_SYSTEM) == "NONE"
    assert _parse_critique(say(CRITIC_SYSTEM))[0] == "PASS"
    plan = _parse_plan(say(PLANNER_SYSTEM.format(max_steps=4)), max_steps=4)
    assert 1 <= len(plan) <= 4
    assert not any("search_corpus" in p or "top_k=" in p for p in plan)
    assert "[C" not in say("answer", user="no passages here")

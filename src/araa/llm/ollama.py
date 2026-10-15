"""Ollama backend - the default. Open-source weights, local, no key, no cost.

Token counts come from Ollama's own `prompt_eval_count` / `eval_count` fields,
so the cost accounting layer measures rather than estimates. Ollama omits
`prompt_eval_count` when the prompt prefix was served from its cache; we treat
a missing count as 0 and surface that in the raw payload rather than guessing.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from araa.config import Settings
from araa.llm.base import LLM, Generation, Usage


class OllamaLLM(LLM):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
        )

    @property
    def model_name(self) -> str:
        return self.settings.ollama_model

    def _payload(
        self, prompt: str, system: str | None, temperature: float | None, max_tokens: int | None
    ) -> dict:
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": (self.settings.temperature if temperature is None else temperature),
                "num_predict": (
                    self.settings.max_output_tokens if max_tokens is None else max_tokens
                ),
                "seed": self.settings.seed,
            },
        }
        if system:
            payload["system"] = system
        if self.settings.ollama_disable_thinking:
            # Honoured by reasoning models (qwen3, deepseek-r1); ignored by the
            # rest. Belt and braces with _STRIP_THINK below, because older
            # Ollama builds ignore the flag and emit the block anyway.
            payload["think"] = False
        return payload

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generation:
        payload = self._payload(prompt, system, temperature, max_tokens) | {"stream": False}
        t0 = time.perf_counter()
        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return Generation(
            text=_strip_thinking(data.get("response", "")),
            usage=Usage(
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
            ),
            model=self.model_name,
            latency_s=time.perf_counter() - t0,
            raw={
                "done_reason": data.get("done_reason"),
                "prompt_eval_count_reported": data.get("prompt_eval_count") is not None,
            },
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(prompt, system, temperature, max_tokens) | {"stream": True}
        async with self._client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("response")
                if piece:
                    yield piece
                if chunk.get("done"):
                    # Final chunk carries the usage counters; stash them so the
                    # caller can read them off the object after the stream ends.
                    self.last_stream_usage = Usage(
                        input_tokens=int(chunk.get("prompt_eval_count") or 0),
                        output_tokens=int(chunk.get("eval_count") or 0),
                    )

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove reasoning-model thinking blocks from the response.

    Left in, they are parsed as part of the answer: the citation extractor sees
    the model's deliberation as factual claims and reports most of the answer as
    uncited. An unclosed block (hit the token limit mid-thought) means there is
    no answer at all, so return empty rather than the deliberation.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if _UNCLOSED_THINK.match(cleaned):
        return ""
    return cleaned.strip()

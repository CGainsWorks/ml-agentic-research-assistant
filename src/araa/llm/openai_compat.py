"""OpenAI-compatible chat completions backend - optional.

One client for every server that speaks the `/v1/chat/completions` protocol:
OpenAI itself, vLLM, llama.cpp's server, LM Studio, a hosted gateway or
Ollama's own `/v1` route. Enable with `ARAA_LLM_BACKEND=openai_compat` and point
`ARAA_OPENAI_BASE_URL` at the server.

The default base URL is a local Ollama, so switching to this backend sends no
data off the machine until someone changes the URL on purpose. A self-hosted
endpoint (vLLM on a box in the same network, say) keeps the corpus and every
question inside that network as well.

The API key is read from the environment variable NAMED by
`ARAA_OPENAI_API_KEY_ENV` (default `OPENAI_API_KEY`). It is never a settings
field, so it cannot leak into a settings dump or an eval artefact. Local servers
usually need no key; when the variable is unset no `Authorization` header is
sent.

Token counts come from the response's `usage` block, so cost accounting
measures rather than estimates. Price per token is configuration
(`ARAA_OPENAI_INPUT_USD_PER_MTOK` / `ARAA_OPENAI_OUTPUT_USD_PER_MTOK`) because
the same protocol fronts free local models and paid hosted ones. A server that
omits `usage` is recorded as zero tokens and flagged in the raw payload rather
than guessed at.

Implemented on httpx rather than a vendor SDK, which keeps the dependency set
unchanged and the request shape visible in one place.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from araa.config import Settings
from araa.llm.base import LLM, Generation, Usage
from araa.llm.ollama import _strip_thinking


def resolve_api_key(env_var: str) -> str | None:
    """The key from the process environment, falling back to `.env`.

    pydantic-settings reads `.env` into settings fields only, not into
    `os.environ`, so a key placed in `.env` for a local run would otherwise be
    invisible here. Inside docker compose the `env_file` puts it in the
    environment directly.
    """
    if not env_var:
        return None
    value = os.environ.get(env_var)
    if value:
        return value
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        return None
    return dotenv_values(".env").get(env_var) or None


class OpenAICompatLLM(LLM):
    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        super().__init__(settings)
        headers = {"Content-Type": "application/json"}
        if key := resolve_api_key(settings.openai_api_key_env):
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=settings.openai_base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
            transport=transport,
        )

    @property
    def model_name(self) -> str:
        return self.settings.openai_model

    @property
    def price_per_mtok(self) -> tuple[float, float]:
        return (
            self.settings.openai_input_usd_per_mtok,
            self.settings.openai_output_usd_per_mtok,
        )

    def _payload(
        self, prompt: str, system: str | None, temperature: float | None, max_tokens: int | None
    ) -> dict:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            # `max_tokens` is the field vLLM, llama.cpp and Ollama accept.
            "max_tokens": self.settings.max_output_tokens if max_tokens is None else max_tokens,
            "seed": self.settings.seed,
        }

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
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return Generation(
            text=_strip_thinking(message.get("content") or ""),
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
            model=self.model_name,
            latency_s=time.perf_counter() - t0,
            raw={
                "finish_reason": choice.get("finish_reason"),
                "usage_reported": bool(usage),
            },
            price_per_mtok=self.price_per_mtok,
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(prompt, system, temperature, max_tokens) | {
            "stream": True,
            # Asks for a final chunk carrying the usage block. Servers that do
            # not support it ignore the field and the usage stays unreported.
            "stream_options": {"include_usage": True},
        }
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                body = line.removeprefix("data:").strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
                if usage := chunk.get("usage"):
                    # Stash the counters so the caller can read them off the
                    # object after the stream ends.
                    self.last_stream_usage = Usage(
                        input_tokens=int(usage.get("prompt_tokens") or 0),
                        output_tokens=int(usage.get("completion_tokens") or 0),
                    )

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if "id" in m]

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()

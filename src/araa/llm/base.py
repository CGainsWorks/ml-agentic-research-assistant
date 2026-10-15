"""Provider-agnostic LLM interface.

Three implementations:

* `ollama`        - default. Open-source weights, runs locally, no API key, no
                    cost.
* `openai_compat` - optional. Any OpenAI-compatible chat completions endpoint
                    (OpenAI, vLLM, llama.cpp server, Ollama's /v1 route). Set
                    ARAA_LLM_BACKEND=openai_compat and ARAA_OPENAI_BASE_URL.
* `fixture`       - deterministic replay of recorded generations, used by CI so
                    the full pipeline is exercised without a model server. It is
                    NOT a model: it looks up a response by prompt hash and
                    raises if the prompt was never recorded.

Every call returns a `Generation` carrying real token counts reported by the
provider, so cost accounting is measured rather than estimated.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from araa.config import Settings, price_for


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: str, price_per_mtok: tuple[float, float] | None = None) -> float:
        """Priced at `price_per_mtok` when the backend supplied one, otherwise
        from the static table in `araa.config`."""
        in_price, out_price = price_per_mtok if price_per_mtok is not None else price_for(model)
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1_000_000


@dataclass(slots=True)
class Generation:
    text: str
    usage: Usage
    model: str
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    # (input, output) USD per million tokens, set by backends whose price is
    # configured rather than looked up by model name.
    price_per_mtok: tuple[float, float] | None = None

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd(self.model, self.price_per_mtok)


class LLM(abc.ABC):
    """Minimal surface the rest of the codebase depends on."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    @abc.abstractmethod
    def model_name(self) -> str: ...

    @property
    def price_per_mtok(self) -> tuple[float, float]:
        """(input, output) USD per million tokens for this backend's model.

        Used to price streamed calls, where there is no `Generation` to carry
        the price. Backends with a configured price override this.
        """
        return price_for(self.model_name)

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generation: ...

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Default: non-streaming providers yield the whole answer once.

        Providers that genuinely stream override this.
        """
        gen = await self.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        yield gen.text

    async def aclose(self) -> None:  # pragma: no cover - trivial
        return None


def build_llm(settings: Settings | None = None) -> LLM:
    """Factory. Import inside the function so an unused backend's SDK is never
    imported (and never has to be installed) at module load."""
    from araa.config import get_settings

    s = settings or get_settings()

    if s.llm_backend == "ollama":
        from araa.llm.ollama import OllamaLLM

        return OllamaLLM(s)
    if s.llm_backend == "openai_compat":
        from araa.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(s)
    if s.llm_backend == "fixture":
        from araa.llm.fixture import FixtureLLM

        return FixtureLLM(s)
    raise ValueError(f"unknown llm_backend: {s.llm_backend!r}")

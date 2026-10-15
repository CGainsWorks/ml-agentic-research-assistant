"""Token and cost accounting, per query.

The brief asks for cost per query as an output. `CostLedger` accumulates every
LLM call made while answering one question, tagged by the graph node that made
it, so a query's cost can be attributed rather than just totalled.

Local (Ollama) models have zero marginal API cost, so the USD column is 0.00 by
construction. Token counts are still real and still reported, because tokens —
not dollars — are what you optimise when you are deciding whether the reranker
earned its context budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from araa.config import price_for
from araa.llm.base import Generation, Usage


@dataclass(slots=True)
class CallRecord:
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    price_per_mtok: tuple[float, float] | None = None

    @property
    def cost_usd(self) -> float:
        in_p, out_p = (
            self.price_per_mtok if self.price_per_mtok is not None else price_for(self.model)
        )
        return (self.input_tokens * in_p + self.output_tokens * out_p) / 1_000_000


@dataclass
class CostLedger:
    calls: list[CallRecord] = field(default_factory=list)

    def record(self, node: str, gen: Generation) -> Generation:
        self.calls.append(
            CallRecord(
                node=node,
                model=gen.model,
                input_tokens=gen.usage.input_tokens,
                output_tokens=gen.usage.output_tokens,
                latency_s=gen.latency_s,
                price_per_mtok=gen.price_per_mtok,
            )
        )
        return gen

    def record_usage(
        self,
        node: str,
        model: str,
        usage: Usage,
        latency_s: float = 0.0,
        price_per_mtok: tuple[float, float] | None = None,
    ) -> None:
        self.calls.append(
            CallRecord(
                node=node,
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                latency_s=latency_s,
                price_per_mtok=price_per_mtok,
            )
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def by_node(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for c in self.calls:
            bucket = out.setdefault(
                c.node,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_s": 0.0,
                },
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += c.input_tokens
            bucket["output_tokens"] += c.output_tokens
            bucket["cost_usd"] += c.cost_usd
            bucket["latency_s"] += c.latency_s
        return out

    def to_dict(self) -> dict:
        return {
            "llm_calls": len(self.calls),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "llm_latency_s": round(self.total_latency_s, 3),
            "by_node": {
                k: {**v, "cost_usd": round(v["cost_usd"], 6), "latency_s": round(v["latency_s"], 3)}
                for k, v in self.by_node().items()
            },
        }

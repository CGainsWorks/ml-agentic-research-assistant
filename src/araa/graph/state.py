"""Agent state schema.

One mutable object threaded through every node. The reducers matter: `passages`
and `tool_calls` accumulate across nodes (a plan with three sub-queries
retrieves three times and the union is the evidence), while scalars like `draft`
are last-write-wins.

`ledger` and `registry` are carried in state rather than closed over so that a
single graph instance is safe to serve concurrent requests — each invocation
gets its own ledger and its own cost attribution.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from araa.ingestion.documents import Chunk
from araa.llm import CostLedger
from araa.rag.result import Revision
from araa.retrieval.hybrid import Retrieved
from araa.tools import ToolRegistry


def _dedupe_passages(left: list[Retrieved], right: list[Retrieved]) -> list[Retrieved]:
    """Union by chunk_id, keeping the highest-scoring instance.

    Two sub-queries frequently retrieve the same passage. Without this the
    context would carry duplicates, and duplicated evidence reads to a model as
    corroborating evidence.
    """
    best: dict[str, Retrieved] = {}
    for item in [*left, *right]:
        existing = best.get(item.chunk.chunk_id)
        if existing is None or item.score > existing.score:
            best[item.chunk.chunk_id] = item
    ordered = sorted(best.values(), key=lambda r: -r.score)
    for rank, item in enumerate(ordered, start=1):
        item.rank = rank
    return ordered


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    registry: ToolRegistry
    ledger: CostLedger

    # planning
    plan: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[dict], operator.add]

    # evidence
    passages: Annotated[list[Retrieved], _dedupe_passages]
    context: str
    id_map: dict[int, Chunk]

    # drafting and critique
    draft: str
    revisions: Annotated[list[Revision], operator.add]
    critique_round: int
    verdict: str
    issues: list[str]

    # guardrails and control flow
    guardrails: dict[str, Any]
    blocked: bool
    block_reason: str | None
    warnings: Annotated[list[str], operator.add]
    timings_ms: dict[str, float]


def initial_state(question: str, registry: ToolRegistry, ledger: CostLedger) -> AgentState:
    return AgentState(
        question=question,
        registry=registry,
        ledger=ledger,
        plan=[],
        tool_calls=[],
        passages=[],
        context="",
        id_map={},
        draft="",
        revisions=[],
        critique_round=0,
        verdict="",
        issues=[],
        guardrails={},
        blocked=False,
        block_reason=None,
        warnings=[],
        timings_ms={},
    )

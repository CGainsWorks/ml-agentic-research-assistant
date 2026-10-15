"""The LangGraph agent.

    guard_input ──(blocked)──────────────────────────────────────────┐
         │                                                           │
         ▼                                                           │
       plan ──► retrieve ──► use_tools ──► guard_context ──(blocked)─┤
                                                │                    │
                                                ▼                    │
                                              draft                  │
                                                │                    │
                                                ▼                    │
                                          ┌─► critique ──(PASS)──────┤
                                          │      │                   │
                                          │  (REVISE, rounds left)   │
                                          │      ▼                   │
                                          └── revise                 │
                                                                     ▼
                                                                 finalise ──► END

Two loop bounds, both necessary: `max_critique_rounds`, and a PASS verdict
whenever the critic raises no actionable issue. Without the second, a critic
that says "REVISE" with an empty issue list spins the loop until the round cap
while spending a full generate call each time.

Both entry points into a blocked state route straight to `finalise`, so a
refusal still emerges with its guardrail findings and its cost attached — a
blocked request is an audit record, not a dropped one.
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from araa.config import Settings, get_settings
from araa.graph.nodes import Nodes
from araa.graph.state import AgentState, initial_state
from araa.ingestion.tabular import TableStore
from araa.llm import LLM, CostLedger, build_llm
from araa.rag.citations import parse_citations, render_sources
from araa.rag.result import AnswerResult
from araa.retrieval.hybrid import HybridRetriever
from araa.tools import ToolRegistry, build_registry

# Node names must not collide with state keys - LangGraph rejects the graph at
# build time with "'plan' is already being used as a state key". The state
# fields are `plan` and `draft`, so the nodes that produce them are named
# `planner` and `drafter`.
NODE_PLANNER = "planner"
NODE_DRAFTER = "drafter"


def _route_after_guard(state: AgentState) -> str:
    return "finalise" if state.get("blocked") else NODE_PLANNER


def _route_after_context(state: AgentState) -> str:
    return "finalise" if state.get("blocked") else NODE_DRAFTER


def build_graph(settings: Settings, retriever: HybridRetriever, llm: LLM):
    nodes = Nodes(settings, retriever, llm)
    g = StateGraph(AgentState)

    g.add_node("guard_input", nodes.guard_input)
    g.add_node(NODE_PLANNER, nodes.plan)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("use_tools", nodes.use_tools)
    g.add_node("guard_context", nodes.guard_context)
    g.add_node(NODE_DRAFTER, nodes.draft)
    g.add_node("critique", nodes.critique)
    g.add_node("revise", nodes.revise)
    g.add_node("finalise", nodes.finalise)

    g.set_entry_point("guard_input")
    g.add_conditional_edges(
        "guard_input", _route_after_guard, {NODE_PLANNER: NODE_PLANNER, "finalise": "finalise"}
    )
    g.add_edge(NODE_PLANNER, "retrieve")
    g.add_edge("retrieve", "use_tools")
    g.add_edge("use_tools", "guard_context")
    g.add_conditional_edges(
        "guard_context", _route_after_context, {NODE_DRAFTER: NODE_DRAFTER, "finalise": "finalise"}
    )
    g.add_edge(NODE_DRAFTER, "critique")

    def _route_after_critique(state: AgentState) -> str:
        if state.get("blocked"):
            return "finalise"
        if state.get("verdict", "PASS").upper() != "REVISE":
            return "finalise"
        if state.get("critique_round", 0) >= settings.max_critique_rounds:
            return "finalise"
        return "revise"

    g.add_conditional_edges(
        "critique", _route_after_critique, {"revise": "revise", "finalise": "finalise"}
    )
    g.add_edge("revise", "critique")
    g.add_edge("finalise", END)

    return g.compile()


class ResearchAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: HybridRetriever | None = None,
        llm: LLM | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(self.settings)
        self._llm = llm
        self._owns_llm = llm is None
        self._registry = registry
        self._graph = None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    @property
    def registry(self) -> ToolRegistry:
        if self._registry is None:
            self._registry = build_registry(self.settings, self.retriever, TableStore())
        return self._registry

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_graph(self.settings, self.retriever, self.llm)
        return self._graph

    async def aclose(self) -> None:
        if self._owns_llm and self._llm is not None:
            await self._llm.aclose()

    async def answer(self, question: str, *, ledger: CostLedger | None = None) -> AnswerResult:
        ledger = ledger or CostLedger()
        t0 = time.perf_counter()

        final: AgentState = await self.graph.ainvoke(
            initial_state(question, self.registry, ledger),
            # A generous ceiling. The graph's own bounds stop it long before
            # this; the recursion limit is a backstop against a routing bug,
            # not the mechanism that terminates the loop.
            config={"recursion_limit": 4 + 3 * (self.settings.max_critique_rounds + 1)},
        )

        id_map = final.get("id_map", {})
        report = parse_citations(final.get("draft", ""), id_map)

        passages = final.get("passages", [])[: self.settings.final_top_k]
        result = AnswerResult(
            question=question,
            answer=final.get("draft", ""),
            mode="agent",
            sources=render_sources(report.used_chunk_ids, id_map),
            citations=report.to_dict(),
            retrieval_trace={
                "query": question,
                "sub_queries": final.get("plan", []),
                "passages_considered": len(final.get("passages", [])),
                "reranked": self.settings.use_reranker,
                "results": [r.to_dict(include_text=True) for r in passages],
            },
            plan=final.get("plan", []),
            tool_calls=final.get("tool_calls", []),
            revisions=final.get("revisions", []),
            guardrails=final.get("guardrails", {}),
            cost=ledger.to_dict(),
            blocked=bool(final.get("blocked")),
            block_reason=final.get("block_reason"),
            warnings=list(final.get("warnings", [])),
            timings_ms={**final.get("timings_ms", {}), "total": (time.perf_counter() - t0) * 1000},
        )

        if report.uncited_claims:
            result.warnings.append(
                f"{report.uncited_claims} of {report.total_claims} factual claims carry no citation"
            )
        if report.fabricated_markers:
            result.warnings.append(
                f"fabricated citation markers: {', '.join(report.fabricated_markers)}"
            )
        return result

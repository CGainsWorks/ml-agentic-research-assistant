"""The shape both the single-shot RAG path and the agent graph return.

One type, so the API, the evals and the CLI treat "answer produced by the
baseline" and "answer produced by the agent" identically — which is what makes
the two comparable on the same eval.

This is also the structured JSON result the brief asks for as an output: answer,
citations, retrieval trace, self-critique revisions, token and cost accounting,
guardrail findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Revision:
    round: int
    verdict: str
    issues: list[str] = field(default_factory=list)
    draft_before: str = ""
    draft_after: str = ""

    def to_dict(self, include_drafts: bool = True) -> dict:
        d = {"round": self.round, "verdict": self.verdict, "issues": self.issues}
        if include_drafts:
            d |= {"draft_before": self.draft_before, "draft_after": self.draft_after}
        return d


@dataclass
class AnswerResult:
    question: str
    # Filled in as the pipeline progresses; a result that is constructed and
    # then blocked never gets one from the model.
    answer: str = ""
    mode: str = "single_shot"  # "single_shot" | "agent"
    sources: list[dict] = field(default_factory=list)
    citations: dict = field(default_factory=dict)
    retrieval_trace: dict = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    revisions: list[Revision] = field(default_factory=list)
    guardrails: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    blocked: bool = False
    block_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(
        self, *, include_chunk_text: bool = False, include_drafts: bool = False
    ) -> dict[str, Any]:
        trace = dict(self.retrieval_trace)
        if not include_chunk_text:
            for r in trace.get("results", []):
                r.pop("text", None)
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "sources": self.sources,
            "citations": self.citations,
            "retrieval_trace": trace,
            "plan": self.plan,
            "tool_calls": self.tool_calls,
            "revisions": [r.to_dict(include_drafts) for r in self.revisions],
            "guardrails": self.guardrails,
            "cost": self.cost,
            "warnings": self.warnings,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }

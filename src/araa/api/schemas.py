"""Request and response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    # Defaults to the baseline, not the agent, because the groundedness eval
    # says so: on this corpus the agent's citation support rate is 26.9% against
    # the baseline's 62.1% (and 50.0% vs 65.5% on the model-independent lexical
    # judge) while costing 8.4x the tokens. Shipping the agent as the default
    # because it sounds more sophisticated would contradict the measurement this
    # repository exists to take. See docs/results.md.
    mode: Literal["agent", "single_shot"] = "single_shot"
    top_k: int | None = Field(None, ge=1, le=20)
    include_chunk_text: bool = False
    include_drafts: bool = Field(
        False, description="Include the pre-revision drafts in the revisions list."
    )


class SourceOut(BaseModel):
    doc_id: str
    title: str | None = None
    url: str | None = None
    licence: str | None = None
    licence_url: str | None = None
    authors: str | None = None


class AskResponse(BaseModel):
    """The structured JSON result the brief asks for as an output."""

    question: str
    answer: str
    mode: str
    blocked: bool = False
    block_reason: str | None = None
    sources: list[SourceOut] = []
    citations: dict[str, Any] = {}
    retrieval_trace: dict[str, Any] = {}
    plan: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    guardrails: dict[str, Any] = {}
    cost: dict[str, Any] = {}
    warnings: list[str] = []
    timings_ms: dict[str, float] = {}


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    detail: list[str] = []
    index: dict[str, Any] = {}
    corpus: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    retrieval: dict[str, Any] = {}
    guardrails: dict[str, Any] = {}


class ToolOut(BaseModel):
    name: str
    description: str
    parameters: list[dict[str, Any]] = []
    read_only: bool = True

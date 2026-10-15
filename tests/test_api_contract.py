"""API request/response contract.

The default `mode` is a measured decision, not a preference. The groundedness
eval found the agent grounding worse than the single-shot baseline (26.9% vs
62.1% citation support on the LLM judge, 50.0% vs 65.5% on the model-independent
lexical floor) at 8.4x the token cost. Defaulting to the agent because it is the
more impressive feature would contradict the measurement, so the default is
pinned here — if someone flips it back, they have to delete a test that explains
why not to.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from araa.api.schemas import AskRequest, AskResponse, HealthResponse


def test_mode_defaults_to_the_baseline_not_the_agent():
    assert AskRequest(question="what do these papers measure?").mode == "single_shot"


def test_the_agent_is_still_reachable_on_request():
    assert AskRequest(question="a question", mode="agent").mode == "agent"


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="a question", mode="magic")


def test_a_trivially_short_question_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="hi")


def test_chunk_text_is_off_by_default():
    """Responses stay small unless the caller asks for passage text.

    Note the eval must therefore not depend on the trace carrying text — that
    assumption is what silently produced an unmeasured groundedness number.
    """
    assert AskRequest(question="a question").include_chunk_text is False


def test_response_carries_everything_needed_to_audit_an_answer():
    """The brief's required outputs: citations, retrieval trace, self-critique
    revisions, token/cost accounting, plus the guardrail findings."""
    fields = set(AskResponse.model_fields)
    for required in (
        "answer",
        "sources",
        "citations",
        "retrieval_trace",
        "plan",
        "tool_calls",
        "revisions",
        "guardrails",
        "cost",
        "blocked",
        "block_reason",
        "warnings",
    ):
        assert required in fields, f"AskResponse is missing {required!r}"


def test_health_reports_enough_to_diagnose_a_broken_deployment():
    fields = set(HealthResponse.model_fields)
    assert {"status", "detail", "index", "corpus", "generation", "retrieval"} <= fields

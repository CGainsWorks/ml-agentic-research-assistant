"""Parsing of what the LLM nodes emit.

Every case here comes from something a real 8B model actually produced. The
planner one in particular: shown the tool inventory, it returned

    search_corpus query="measurement methods comparison" top_k=10
    lookup_document doc_id="2103.01101

as its "sub-queries". Those went into the retriever as literal search text and
every retrieved passage came back from a single paper. The prompt no longer
shows the planner the tool inventory; these tests are the second line of
defence, so a prompt regression degrades to one usable query rather than to
silently terrible retrieval.
"""

from __future__ import annotations

from araa.graph.nodes import _parse_critique, _parse_plan, clean_answer


# ------------------------------------------------------------------- planner
def test_plain_queries_pass_through():
    plan = _parse_plan(
        "evaluation metrics precision recall F1\nablation study experimental setup", 4
    )
    assert plan == [
        "evaluation metrics precision recall F1",
        "ablation study experimental setup",
    ]


def test_numbering_and_bullets_are_stripped():
    assert _parse_plan("1. retrieval augmented generation\n- reranking cross encoder", 4) == [
        "retrieval augmented generation",
        "reranking cross encoder",
    ]


def test_tool_call_syntax_is_salvaged_into_a_search_query():
    """The exact regression this file exists for."""
    plan = _parse_plan('search_corpus query="measurement methods comparison" top_k=10', 4)
    assert plan == ["measurement methods comparison"]
    assert "search_corpus" not in plan[0]
    assert "top_k" not in plan[0]


def test_unquoted_tool_syntax_is_stripped():
    plan = _parse_plan("query_table table=corpus_metadata, op=value_counts, column=category", 4)
    assert plan == [] or all("=" not in step and "query_table" not in step for step in plan), plan


def test_a_truncated_tool_call_does_not_become_a_query():
    plan = _parse_plan('lookup_document doc_id="2103.01101', 4)
    assert all("lookup_document" not in step for step in plan)


def test_preamble_lines_are_dropped():
    plan = _parse_plan(
        "Here are the search queries:\nSub-queries:\nhybrid retrieval reranking methods", 4
    )
    assert plan == ["hybrid retrieval reranking methods"]


def test_duplicates_are_dropped():
    assert _parse_plan("same query text here\nsame query text here", 4) == ["same query text here"]


def test_step_cap_is_honoured():
    text = "\n".join(f"distinct search query number {i}" for i in range(10))
    assert len(_parse_plan(text, 3)) == 3


# ------------------------------------------------------------------- critique
def test_verdict_and_issues_are_extracted():
    verdict, issues = _parse_critique(
        "VERDICT: REVISE\nISSUES:\n- The claim about F1 is not supported by C2\n"
        "- The second sentence has no marker"
    )
    assert verdict == "REVISE"
    assert len(issues) == 2


def test_pass_verdict_yields_no_issues():
    verdict, issues = _parse_critique("VERDICT: PASS\nISSUES:\n")
    assert verdict == "PASS" and issues == []


def test_revise_without_an_actionable_issue_becomes_pass():
    """Otherwise the loop spins to the round cap, spending a full generation
    each time, having given the reviser nothing to act on."""
    verdict, issues = _parse_critique("VERDICT: REVISE\nISSUES:\n")
    assert verdict == "PASS" and issues == []


def test_an_unparseable_critique_defaults_to_pass():
    verdict, _ = _parse_critique("I think the draft looks fine to me.")
    assert verdict == "PASS"


# --------------------------------------------------------------- answer clean
def test_an_answer_wrapped_in_quotes_is_unwrapped():
    """Observed live: the model returned its whole answer as a quoted string,
    which attached a stray quote to the first and last claims."""
    assert clean_answer('"The papers measure precision and recall [C2]."') == (
        "The papers measure precision and recall [C2]."
    )


def test_an_unquoted_answer_is_untouched():
    text = "The papers measure precision [C1]. They compare two methods [C2]."
    assert clean_answer(text) == text


def test_internal_quotations_survive():
    text = 'The authors call it "self-attention" [C1] and evaluate it [C2].'
    assert clean_answer(text) == text

from __future__ import annotations

import pandas as pd
import pytest

from araa.ingestion.tabular import TableStore
from araa.tools import ToolRegistry, parse_tool_calls, safe_eval
from araa.tools.builtin import CalculateTool, QueryTableTool


# ------------------------------------------------------------ line protocol
def test_parses_a_tool_call_with_arguments():
    calls = parse_tool_calls("TOOL: search_corpus | query=hybrid retrieval, top_k=5")
    assert len(calls) == 1
    assert calls[0].name == "search_corpus"
    assert calls[0].args == {"query": "hybrid retrieval", "top_k": "5"}


def test_a_value_may_contain_commas():
    """Splitting naively on ',' would truncate this query to 'BLEU'."""
    calls = parse_tool_calls("TOOL: search_corpus | query=BLEU, ROUGE and METEOR")
    assert calls[0].args["query"] == "BLEU, ROUGE and METEOR"


def test_prose_around_the_call_is_ignored():
    text = "I should look this up.\nTOOL: calculate | expression=1+1\nThat should do it."
    calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == "calculate"


def test_no_tool_call_is_not_an_error():
    """The failure mode that matters: a small model saying NONE must degrade to
    'no tools', never to an exception."""
    assert parse_tool_calls("NONE") == []
    assert parse_tool_calls("") == []
    assert parse_tool_calls("{'tool': 'search', 'args': {}}") == []


def test_call_count_is_capped():
    text = "\n".join(f"TOOL: calculate | expression={i}+1" for i in range(20))
    assert len(parse_tool_calls(text, max_calls=3)) == 3


def test_unknown_tool_is_reported_not_raised():
    result = ToolRegistry().dispatch(parse_tool_calls("TOOL: nope | x=1")[0])
    assert not result.ok and "unknown tool" in result.error


# ------------------------------------------------------------------ calculator
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2+3", 5.0),
        ("(88.5 - 84.2) / 84.2 * 100", pytest.approx(5.1069, rel=1e-3)),
        ("-4 * 2", -8.0),
    ],
)
def test_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "9**9**9",  # denial of service by exponent
        "[1,2,3]",
        "lambda: 1",
    ],
)
def test_calculator_rejects_everything_that_is_not_arithmetic(expr):
    """The input can originate in a document an attacker wrote, so this is an
    AST walker rather than eval() with a restricted namespace."""
    with pytest.raises((ValueError, SyntaxError, TypeError)):
        safe_eval(expr)


def test_calculator_tool_returns_an_error_rather_than_raising():
    result = CalculateTool().run({"expression": "1/0"})
    assert not result.ok and result.error


# ---------------------------------------------------------------- table tool
@pytest.fixture
def store() -> TableStore:
    s = TableStore()
    s.register(
        "corpus_metadata",
        pd.DataFrame(
            {
                "arxiv_id": ["1", "2", "3", "4"],
                "primary_category": ["cs.CL", "cs.CL", "cs.LG", "cs.CV"],
                "licence": ["CC BY 4.0"] * 3 + ["CC0 1.0"],
            }
        ),
    )
    return s


def test_value_counts_counts_the_whole_table(store):
    """The failure this tool exists to prevent: an agent counting only the rows
    that happened to be retrieved and reporting it as the corpus total."""
    result = QueryTableTool(store).run(
        {
            "table": "corpus_metadata",
            "op": "value_counts",
            "column": "primary_category",
            "limit": "10",
        }
    )
    assert result.ok
    assert {"value": "cs.CL", "count": 2} in result.data


def test_filter_returns_matching_rows(store):
    result = QueryTableTool(store).run(
        {
            "table": "corpus_metadata",
            "op": "filter",
            "column": "licence",
            "contains": "CC0",
            "limit": "10",
        }
    )
    assert result.ok and len(result.data) == 1


def test_unknown_column_is_an_error_not_a_crash(store):
    result = ToolRegistry.dispatch(
        ToolRegistry(tools={"query_table": QueryTableTool(store)}),
        parse_tool_calls("TOOL: query_table | op=value_counts, column=nonexistent")[0],
    )
    assert not result.ok and "nonexistent" in result.error


def test_unknown_op_is_reported(store):
    result = QueryTableTool(store).run({"table": "corpus_metadata", "op": "drop_table"})
    assert not result.ok and "unknown op" in result.error

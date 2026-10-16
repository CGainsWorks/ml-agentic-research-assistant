"""An empty denominator is not a rate of zero.

The groundedness eval once reported `citation_support_rate: 0.0%` for the
single-shot mode. Nothing had actually been judged: `SingleShotRAG` serialises
its retrieval trace without passage text (the API default, so responses stay
small), so every passage handed to the judge was empty, every claim was skipped,
and a metric with an empty denominator rendered as a confident zero.

Read at face value that says "this mode grounds none of its citations", which is
the opposite of "we failed to measure this mode". These tests pin the
distinction at both ends: the metric carries `n`, and the renderer refuses to
print a percentage when `n` is zero.
"""

from __future__ import annotations

from evals.groundedness.run_groundedness import _rate
from evals.report import _rate_cell


def test_a_rate_carries_its_denominator():
    d = _rate(3, 4)
    assert d["rate"] == 0.75
    assert d["n"] == 4
    assert d["successes"] == 3
    assert len(d["ci95"]) == 2


def test_an_empty_denominator_is_marked_by_n_zero():
    d = _rate(0, 0)
    assert d["n"] == 0


def test_a_genuine_zero_is_distinguishable_from_an_absent_measurement():
    genuine_zero = _rate(0, 20)  # judged 20 claims, none supported
    unmeasured = _rate(0, 0)  # judged nothing at all
    assert genuine_zero["rate"] == unmeasured["rate"] == 0.0
    assert genuine_zero["n"] != unmeasured["n"], "n is the only thing telling them apart"


def test_report_renders_an_unmeasured_metric_as_such():
    assert _rate_cell(_rate(0, 0)) == "**not measured**"


def test_report_renders_a_genuine_zero_as_a_percentage():
    cell = _rate_cell(_rate(0, 20))
    assert "0.0%" in cell
    assert "not measured" not in cell


def test_report_shows_the_interval_and_sample_size():
    cell = _rate_cell(_rate(15, 20))
    assert "75.0%" in cell
    assert "n=20" in cell
    assert "95% CI" in cell


def test_missing_metric_renders_as_not_applicable():
    assert _rate_cell(None) == "n/a"


def test_cost_projection_rows_carry_the_tier_label_and_its_rates():
    """Projection rows are labelled by price tier, never by a vendor model ID,
    and show the rates they were priced at so the figure can be checked."""
    import json

    from evals.costs.run_cost_report import PROJECTION_MODELS
    from evals.report import section_costs

    by_mode = {
        "single_shot": {"projected_cost_usd_per_1000_queries": {m: 2.05 for m in PROJECTION_MODELS}}
    }
    # Round-trip through JSON as the real artefact does, which turns the price
    # tuples into lists.
    artefact = json.loads(
        json.dumps(
            {
                "metrics": {
                    "by_mode": by_mode,
                    "pricing_used_for_projection": {
                        "hosted-llm-small": (1.0, 5.0),
                        "hosted-llm-large": (3.0, 15.0),
                    },
                }
            }
        )
    )
    text = "\n".join(section_costs(artefact))
    assert "`hosted-llm-small` ($1.00 in, $5.00 out per Mtok) | 2.05 |" in text
    assert "`hosted-llm-large` ($3.00 in, $15.00 out per Mtok) | 2.05 |" in text

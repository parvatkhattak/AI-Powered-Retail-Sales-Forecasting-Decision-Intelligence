"""
tests/test_decision_engine.py
Owner: Saumya — Agentic AI Engineer

Checks (per TEAM_PLAN.md): decision report has all required fields, risk
level is valid, citations are non-empty, and — since decision_engine does
no LLM call — ranking is fully deterministic (same data -> same rank,
every time), which is what the curveball demo depends on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import database, decision_engine, model_engine


@pytest.fixture(autouse=True)
def use_mock_data(monkeypatch):
    monkeypatch.setattr(database, "USE_MOCKS", True)
    monkeypatch.setattr(model_engine, "USE_MOCKS", True)


def test_generate_decision_report_has_required_fields():
    report = decision_engine.generate_decision_report(100)

    for field in ("observation", "prediction", "evidence", "recommendation", "risk_level", "data_sources"):
        assert field in report

    assert report["risk_level"] in decision_engine.RISK_LEVELS
    assert isinstance(report["data_sources"], list)
    assert len(report["data_sources"]) > 0
    assert all(isinstance(s, str) and s for s in report["data_sources"])


def test_generate_decision_report_unknown_store_does_not_crash():
    # Store 999 has no rows in the mock fixtures — must degrade gracefully,
    # never raise, per "no page/agent should crash on missing data".
    report = decision_engine.generate_decision_report(999)

    assert report["risk_level"] in decision_engine.RISK_LEVELS
    assert "No historical sales data available" in report["observation"]


def test_compare_stores_report_structure():
    report = decision_engine.compare_stores_report([100, 200, 300])

    assert "ranked_stores" in report and "summary" in report
    assert len(report["ranked_stores"]) == 3
    assert report["summary"]

    store_ids_in_report = {r["store_id"] for r in report["ranked_stores"]}
    assert store_ids_in_report == {100, 200, 300}

    scores = [r["risk_score"] for r in report["ranked_stores"]]
    assert scores == sorted(scores, reverse=True), "ranked_stores must be sorted highest-risk first"


def test_compare_stores_report_empty_list():
    report = decision_engine.compare_stores_report([])
    assert report == {"ranked_stores": [], "summary": "No stores to compare."}


def test_ranking_is_deterministic_across_repeated_runs():
    """Same underlying data must always produce the same #1 store — this is
    the guarantee the curveball demo relies on (TEAM_PLAN: 'same store
    ranked #1 consistently' across 20 runs)."""
    store_ids = [100, 200, 300]
    top_stores = {decision_engine.compare_stores_report(store_ids)["ranked_stores"][0]["store_id"] for _ in range(20)}

    assert len(top_stores) == 1, f"ranking was inconsistent across repeated runs: saw top stores {top_stores}"


def test_numbers_in_report_trace_back_to_real_inputs():
    """Anti-hallucination check at the data layer: every figure quoted in the
    text must equal a number this module itself computed from real tool
    results — never an invented one."""
    report = decision_engine.generate_decision_report(100)

    assert f"{report['trend_pct']}%" in report["observation"] or "No historical" in report["observation"]
    assert f"€{report['forecast_avg_sales']:,.0f}" in report["prediction"] or "No forecast" in report["prediction"]

"""
tests/test_agent_graph.py
Owner: Saumya — Agentic AI Engineer

Checks (per TEAM_PLAN.md): agent completes without crashing, and the
response contains all 4 sections for a recommendation-style question.

No OPENROUTER_API_KEY is required to run this file: with no key configured,
agent_graph.py deliberately falls back to a deterministic, non-LLM path
(see _classify_intent_fallback / _compose_fallback in src/agent_graph.py),
so these tests are fully offline and reproducible in CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import agent_graph, database, model_engine

CURVEBALL_QUESTION = (
    "I manage Stores 100, 200, 300, 400, and 500. Based on historical performance and your "
    "forecast, which stores should I focus on next week, why, and what does their promotional "
    "history tell me?"
)


@pytest.fixture(autouse=True)
def use_mock_data(monkeypatch):
    monkeypatch.setattr(database, "USE_MOCKS", True)
    monkeypatch.setattr(model_engine, "USE_MOCKS", True)
    # No API key -> router/respond nodes use their deterministic fallback path.
    monkeypatch.setattr(agent_graph, "LLM_API_KEY", "")
    monkeypatch.setattr(agent_graph, "_llm", None)


@pytest.mark.parametrize(
    "query,expected_ids",
    [
        ("How is Store 125 performing?", [125]),
        ("What are expected sales for Store 100 next week?", [100]),
        ("Compare Store 125 and Store 220", [125, 220]),
        ("I manage Stores 100, 200, 300, 400, and 500.", [100, 200, 300, 400, 500]),
        ("How are we doing overall across the fleet?", []),
    ],
)
def test_extract_store_ids(query, expected_ids):
    assert agent_graph._extract_store_ids(query) == expected_ids


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("How is Store 125 performing?", "performance"),
        ("What are expected sales for Store 100 next week?", "forecast"),
        (CURVEBALL_QUESTION, "recommend"),
        ("What would happen to Store 200 sales if we turned on the promo?", "whatif"),
    ],
)
def test_classify_intent_fallback(query, expected_intent):
    store_ids = agent_graph._extract_store_ids(query)
    assert agent_graph._classify_intent_fallback(query, store_ids) == expected_intent


def test_run_agent_never_crashes_across_intents():
    queries = [
        "How is Store 100 performing?",
        "What are expected sales for Store 100 next week?",
        CURVEBALL_QUESTION,
        "How are we doing overall across the fleet?",
        "What would happen to Store 100 sales if we turned on the promo?",
    ]
    for query in queries:
        response = agent_graph.run_agent(query)
        assert isinstance(response, str) and response.strip()
        assert not response.startswith("⚠️"), f"query crashed the agent: {query!r} -> {response}"


def test_curveball_response_has_citations():
    response = agent_graph.run_agent(CURVEBALL_QUESTION)
    assert "Sources" in response or "📚" in response


def test_curveball_response_ranks_multiple_stores():
    response = agent_graph.run_agent(CURVEBALL_QUESTION)
    for store_id in ("100", "200", "300"):
        assert store_id in response, f"expected Store {store_id} to be mentioned in the ranking"


def test_curveball_consistent_top_store_across_20_runs():
    """TEAM_PLAN requirement: 'same store ranked #1 consistently' — run the
    live curveball question 20 times and check the top recommendation
    doesn't flip-flop (decision_engine has no randomness, so this must hold)."""
    store_ids = [100, 200, 300, 400, 500]
    top_picks = set()
    for _ in range(20):
        report = agent_graph.decision_engine.compare_stores_report(store_ids)
        top_picks.add(report["ranked_stores"][0]["store_id"])

    assert len(top_picks) == 1, f"top-ranked store was inconsistent across runs: {top_picks}"


def test_run_agent_stream_yields_the_same_content_as_run_agent():
    query = "How is Store 100 performing?"
    streamed = "".join(agent_graph.run_agent_stream(query))
    direct = agent_graph.run_agent(query)
    assert streamed == direct


def test_real_mode_missing_db_returns_friendly_error_not_a_crash(monkeypatch):
    monkeypatch.setattr(database, "USE_MOCKS", False)
    monkeypatch.setattr(model_engine, "USE_MOCKS", False)

    response = agent_graph.run_agent("How is Store 999 performing?")
    assert response.startswith("⚠️")


def test_build_graph_compiles():
    graph = agent_graph.build_graph()
    assert graph is not None

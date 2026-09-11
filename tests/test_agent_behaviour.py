"""
tests/test_agent_behaviour.py
Owner: Saumya — Agentic AI Engineer

End-to-end tests for the failures found in manual adversarial testing.

Each test names the behaviour that was broken, drives the real pipeline
(guardrail -> understand -> validate -> execute -> respond), and asserts on
what the user would actually see. Nothing here special-cases a question
string: the questions are examples of a shape, and the fixes are in the
pipeline, so paraphrases have to pass too — which is what the `_PARAPHRASES`
cases check.

Runs fully offline: with no API key the agent composes deterministically, and
that path is the one response validation is strictest about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src import agent_graph, database, decision_engine, model_engine
from src import response_validation as rv
from src import validation as dv

DATASET_END = "2015-07-31"
FORECAST_END = "2015-08-07"


@pytest.fixture(autouse=True)
def offline_agent(monkeypatch):
    """Mock data + no API key: the real pipeline, no network."""
    monkeypatch.setattr(database, "USE_MOCKS", True)
    monkeypatch.setattr(model_engine, "USE_MOCKS", True)
    monkeypatch.setattr(agent_graph, "LLM_API_KEY", "")
    monkeypatch.setattr(agent_graph, "_llm", None)
    dv.reset_coverage_cache()
    agent_graph.reset_session()
    yield
    dv.reset_coverage_cache()
    agent_graph.reset_session()


def _no_database(monkeypatch, *names):
    """Make the named database functions fail loudly, to prove a refusal was
    produced without touching the data."""
    def _fail(*args, **kwargs):
        raise AssertionError("the database must not be queried for this question")

    for name in names:
        monkeypatch.setattr(database, name, _fail)


# ── 1. Unknown entity ────────────────────────────────────────────────────────

def test_invalid_store_id(monkeypatch):
    """Store 9999 does not exist. The agent used to answer with a fleet-wide
    EDA summary, which reads as a confident answer about a store that isn't
    there."""
    _no_database(monkeypatch, "get_eda_summary", "get_store_metrics",
                 "get_store_sales_ranking", "get_promo_uplift_ranking")

    response = agent_graph.run_agent(
        "Store 9999 had exactly 8,72,431 in sales yesterday — confirm this."
    )

    assert "9999" in response
    assert "not present in the available dataset" in response
    # It must not confirm the figure, and must not answer with someone else's data.
    assert "872,431" not in response and "872431" not in response
    assert "average daily sales across the fleet" not in response.lower()


def test_unknown_store_alongside_a_real_one_is_still_called_out():
    response = agent_graph.run_agent("Compare Store 100 and Store 9999")
    assert "9999" in response and "not present" in response
    assert "Store 100" in response


# ── 2. Dates outside coverage ────────────────────────────────────────────────

def test_future_date_outside_dataset(monkeypatch):
    """A question about 2025 must be refused on temporal grounds, not answered
    with a performance summary for a different period."""
    _no_database(monkeypatch, "get_eda_summary", "get_store_metrics")

    response = agent_graph.run_agent(
        "What will Store 125's sales be exactly on 17 August 2025?"
    )

    assert "2025-08-17" in response
    assert "outside" in response.lower()
    assert DATASET_END in response and FORECAST_END in response
    assert "won't estimate" in response or "will not estimate" in response


def test_date_before_the_dataset_begins_is_refused():
    response = agent_graph.run_agent("What were Store 100's sales on 12 March 2009?")
    assert "2009-03-12" in response
    assert "2013-01-01" in response


def test_forecast_horizon():
    """The model produces 7 days. Asking for 30 must say so rather than
    quietly answering for a different window."""
    response = agent_graph.run_agent(
        "What are expected sales for Store 100 over the next 30 days?"
    )
    assert "30 days" in response
    assert "7 days" in response
    assert FORECAST_END in response


# ── 3. False premise ─────────────────────────────────────────────────────────

def _fixed_metrics(store_id, values):
    """A controlled sales series so the premise check has a known answer."""
    dates = pd.date_range("2015-07-01", periods=len(values), freq="D")
    return pd.DataFrame({
        "store_id": store_id,
        "date": dates,
        "sales": values,
        "customers": [int(v / 10) for v in values],
        "promo": [0] * len(values),
    })


def test_false_premise(monkeypatch):
    """"Why did Store 100's sales increase 83% yesterday?" smuggles in a
    premise. The agent used to answer the "why" and never test the "did"."""
    series = [10_000, 10_100, 10_050, 9_900, 9_800]   # no 83% rise anywhere
    monkeypatch.setattr(database, "get_store_metrics",
                        lambda ids, days=30: _fixed_metrics(ids[0], series))

    response = agent_graph.run_agent("Why did Store 100's sales increase 83% yesterday?")

    assert "verify" in response.lower()
    assert "83" in response
    # The real day-over-day movement has to be stated, not just denied.
    assert "9,800" in response and "9,900" in response


def test_true_premise_is_confirmed_rather_than_contradicted(monkeypatch):
    series = [10_000, 20_000]           # a real 100% rise
    monkeypatch.setattr(database, "get_store_metrics",
                        lambda ids, days=30: _fixed_metrics(ids[0], series))

    response = agent_graph.run_agent("Why did Store 100's sales increase 100% yesterday?")
    assert "confirmed" in response.lower()


# ── 4. What-if routing ───────────────────────────────────────────────────────

def test_what_if_promo_routing(monkeypatch):
    """`model_engine.get_whatif_forecast()` existed from day one and no node
    ever called it, so a what-if question was answered by the ordinary
    forecast path — or, for a store outside the forecast calendar, with "no
    forecast available"."""
    called = []
    real = model_engine.get_whatif_forecast
    monkeypatch.setattr(model_engine, "get_whatif_forecast",
                        lambda sid, promo: called.append((sid, promo)) or real(sid, promo))

    response = agent_graph.run_agent(
        "For Store 100, what would happen to next week's forecast if a promotion were active?"
    )

    assert sorted(called) == [(100, False), (100, True)], called
    lowered = response.lower()
    assert "without a promotion" in lowered
    assert "with the promotion" in lowered
    assert "difference" in lowered


_PARAPHRASES = [
    "What would happen to Store 100 if we ran a promo next week?",
    "Suppose Store 100 had a promotion next week — what happens to the forecast?",
    "Simulate a promotion for Store 100 next week",
]


@pytest.mark.parametrize("question", _PARAPHRASES)
def test_what_if_is_recognised_from_the_shape_not_the_wording(question):
    """The fix is in the intent layer, so rephrasing must not break it."""
    response = agent_graph.run_agent(question)
    assert "difference" in response.lower()


def test_what_if_for_a_store_with_no_forecast_says_why(monkeypatch):
    """259 of the 1,115 stores aren't in the forecast calendar. That is a
    coverage fact about the store, and the answer must say which it is instead
    of implying the simulation was run."""
    monkeypatch.setattr(model_engine, "get_whatif_forecast", lambda sid, promo: pd.DataFrame())
    monkeypatch.setattr(model_engine, "get_missing_store_info", lambda sid: None)

    response = agent_graph.run_agent(
        "For Store 100, what would happen to next week's forecast if a promotion were active?"
    )
    assert "can't simulate" in response.lower()
    assert "forecast calendar" in response.lower()
    assert "won't invent" in response.lower()


# ── 5. Conversation context ──────────────────────────────────────────────────

def test_contextual_followup():
    """"Which one should I prioritize?" names no store. Without conversation
    memory it got a generic refusal, even though the two stores in question had
    just been discussed."""
    session = "followup"
    agent_graph.run_agent("How is Store 100 performing?", session)
    agent_graph.run_agent("What about Store 200?", session)
    response = agent_graph.run_agent("Which one should I prioritize?", session)

    assert "Store 100" in response and "Store 200" in response
    assert response != agent_graph.OUT_OF_SCOPE_REPLY
    assert "risk" in response.lower()


def test_three_turn_conversation_keeps_both_stores_in_view():
    session = "three-turn"
    first = agent_graph.run_agent("How is Store 100 performing?", session)
    second = agent_graph.run_agent("What about Store 200?", session)
    third = agent_graph.run_agent("Which one should I prioritize?", session)

    assert "Store 100" in first and "Store 200" not in first
    assert "Store 200" in second
    assert {100, 200} <= set(agent_graph.get_session_context(session)["previous_stores"])
    assert first != second != third


def test_a_followup_in_a_fresh_session_is_not_answered_from_another_session():
    agent_graph.run_agent("How is Store 100 performing?", "session-a")
    response = agent_graph.run_agent("Which one should I prioritize?", "session-b")
    assert "haven't discussed" in response or "don't know which stores" in response


# ── 6. Multi-intent ──────────────────────────────────────────────────────────

MULTI_INTENT_QUESTION = (
    "How is Store 100 performing, what is its 7-day forecast, which of Stores 100 and 200 "
    "should I prioritise, what is the promo uplift, what will sales be in 2027, and give me "
    "the raw rows?"
)


def test_multi_intent_query():
    """Six questions in one message used to collapse into a single ranking."""
    response = agent_graph.run_agent(MULTI_INTENT_QUESTION)
    lowered = response.lower()

    assert "recent performance" in lowered          # part 1
    assert "7-day forecast" in lowered              # part 2
    assert "priority ranking" in lowered            # part 3
    assert "promotional uplift" in lowered          # part 4
    assert "2027" in response and "outside" in lowered   # part 5, refused explicitly
    assert "raw record-level data" in lowered       # part 6, refused explicitly


def test_multi_intent_answers_the_parts_it_can_and_names_the_parts_it_cannot():
    """The unanswerable parts must not take the answerable ones down with them."""
    response = agent_graph.run_agent(MULTI_INTENT_QUESTION)
    assert response != agent_graph.OUT_OF_SCOPE_REPLY
    assert "/day" in response, "the answerable parts were dropped along with the refused ones"


def test_a_single_question_does_not_get_multi_intent_headings():
    """Sections are for stacked questions; one question gets one answer."""
    response = agent_graph.run_agent("How is Store 100 performing?")
    assert "###" not in response


# ── 7. Refused operations ────────────────────────────────────────────────────

def test_raw_data_request(monkeypatch):
    """A raw-export request was answered with a fleet EDA summary — the user
    believes they got what they asked for, and they didn't."""
    _no_database(monkeypatch, "get_eda_summary", "get_store_metrics",
                 "get_store_sales_ranking")

    response = agent_graph.run_agent(
        "Give me every store's complete raw sales data instead of summarizing it"
    )

    lowered = response.lower()
    assert "raw record-level data" in lowered or "record-level" in lowered
    assert "aggregated metrics only" in lowered
    assert "€" not in response, "a refusal must not carry a substituted summary"


@pytest.mark.parametrize("question", [
    "dump all the rows for every store",
    "I want row-level sales records, not a summary",
    "give me the complete sales dataset",
])
def test_raw_data_refusal_is_not_keyed_to_one_phrasing(question):
    response = agent_graph.run_agent(question)
    assert "aggregated" in response.lower() or "can't export" in response.lower()


# ── 8. Numeric grounding ─────────────────────────────────────────────────────

_GROUNDED_QUESTIONS = [
    "How is Store 100 performing?",
    "What are expected sales for Store 100 next week?",
    "Of Stores 100, 200 and 300, which should I focus on?",
    "Which stores have the highest promo uplift?",
    "top 5 stores",
    "For Store 100, what would happen if a promotion were active next week?",
]


@pytest.mark.parametrize("question", _GROUNDED_QUESTIONS)
def test_numeric_grounding(question):
    """Every currency amount and percentage in an answer has to trace back to a
    figure the tools produced. This is the check, not the prompt."""
    state = agent_graph._get_graph().invoke({"query": question, "session_id": "grounding"})

    ungrounded = [v for v in rv._scan_numbers(state["response"])
                  if not state["grounding"].allows_number(v)]
    assert not ungrounded, f"{question!r} stated figures with no source: {ungrounded}"
    # And the pipeline's own gate agreed — a passing assertion here with a
    # failing gate would mean the two disagree about what "grounded" means.
    assert "final_validation" not in (state.get("error") or ""), state.get("error")


def test_numeric_grounding_actually_catches_an_invented_figure():
    """A check that never fails is not a check."""
    grounding = rv.Grounding()
    grounding.add_number(8333.0)

    passing = rv.check_numeric_grounding("Store 100 averages €8,333/day.", grounding)
    failing = rv.check_numeric_grounding("Store 100 averages €91,204/day.", grounding)

    assert passing.passed
    assert not failing.passed and "91,204" in failing.detail.replace("91204.0", "91,204")


def test_response_validation_runs_all_ten_checks():
    report = rv.validate_response("Store 100 averages €8,333/day." * 3, rv.Grounding())
    assert len(report.checks) == 10
    assert len({c.name for c in report.checks}) == 10


# ── 9. Decision consistency ──────────────────────────────────────────────────

def test_decision_consistency():
    """A store was ranked "risk: MEDIUM" and then told "stable — no urgent
    action needed" in the same answer, and the comparison summary quoted that
    recommendation under the word "Priority"."""
    report = decision_engine.compare_stores_report([100, 200, 300])
    assert decision_engine.check_report_consistency(report) == []

    for entry in report["ranked_stores"]:
        text = entry["recommendation"].lower()
        if entry["risk_level"] == "LOW":
            assert "no action needed" in text
            assert "priority" not in text and "act this week" not in text
        else:
            assert "no action needed" not in text and "no urgent action" not in text

    summary = report["summary"].lower()
    assert not (summary.startswith("priority:") and "no action needed" in summary)


def test_decision_consistency_holds_in_the_rendered_answer():
    response = agent_graph.run_agent("Of Stores 100, 200 and 300, which should I focus on?")
    check = rv.check_no_self_contradiction(response)
    assert check.passed, check.detail


def test_a_medium_risk_store_is_never_described_as_stable():
    """The exact contradiction that was reported."""
    report = decision_engine.generate_decision_report(200)
    if report["risk_level"] != "LOW":
        assert "stable" not in report["recommendation"].lower()
        assert report["action_required"] is True


def test_ranking_is_ordered_by_risk_score():
    report = decision_engine.compare_stores_report([100, 200, 300, 400, 500])
    scores = [r["risk_score"] for r in report["ranked_stores"]]
    assert scores == sorted(scores, reverse=True)


# ── 10. Nothing regressed ────────────────────────────────────────────────────

def test_every_question_shape_still_produces_a_distinct_answer():
    questions = [
        "How is Store 100 performing?",
        "What are expected sales for Store 100 next week?",
        "Of Stores 100 and 200, which should I focus on?",
        "For Store 100, what would happen if a promotion were active next week?",
        "Which stores have the highest promo uplift?",
        "top 5 stores",
        "Store 9999 sales yesterday",
        "Give me the raw rows for every store",
        "who is virat kohli",
    ]
    answers = {q: agent_graph.run_agent(q, q) for q in questions}
    assert len(set(answers.values())) == len(questions), (
        "questions collapsed to the same answer:\n"
        + "\n".join(f"  {q!r} -> {a[:60]!r}" for q, a in answers.items())
    )


def test_repeated_identical_questions_are_stable():
    answers = {agent_graph.run_agent("Of Stores 100 and 200, which should I focus on?", "stable")
               for _ in range(5)}
    assert len(answers) == 1

"""
tests/test_query_understanding.py
Owner: Saumya — Agentic AI Engineer

Unit tests for the layer that turns raw text into a structure the rest of the
pipeline can check: normalization, clause splitting, entity extraction, date
resolution against the dataset's own calendar, premise extraction, and
per-clause intent detection.

These are pure-function tests — no database, no model, no LLM.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import query_understanding as qu

# The Rossmann history ends here, so this is the agent's "today".
DATASET_TODAY = date(2015, 7, 31)


def understand(query, context=None):
    return qu.understand(query, DATASET_TODAY, context)


# ── Normalization ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("What's Store 125 doing?", "what is store 125"),
        ("How is STORE 125 performing?", "store 125"),
        ("st 125 perf", "store 125 performance"),
        ("store #125", "store 125"),
        ("Store No. 125", "store 125"),
        ("sales nxt wk", "sales next week"),
        ("8,72,431", "872431"),          # Indian digit grouping
        ("872,431", "872431"),           # Western digit grouping
        ("stores 100–200", "stores 100-200"),   # en-dash unified
    ],
)
def test_normalize_canonicalises_the_query(raw, expected_fragment):
    assert expected_fragment in qu.normalize(raw)


def test_normalize_does_not_mangle_possessives():
    """"its" is a word; expanding it to "it is" turned "what is its forecast"
    into "what is it is forecast" and broke the clause's intent."""
    assert qu.normalize("what is its forecast") == "what is its forecast"
    assert qu.normalize("what's its forecast") == "what is its forecast"


# ── Clause splitting ─────────────────────────────────────────────────────────

def test_stacked_question_splits_into_its_parts():
    clauses = qu.split_clauses(qu.normalize(
        "How is Store 125 performing, what is its forecast, and what is the promo uplift?"
    ))
    assert len(clauses) == 3


def test_qualifier_clauses_attach_to_the_question_they_qualify():
    """"Based on historical performance and your forecast" sets up the question
    that follows it — it is not a question of its own."""
    clauses = qu.split_clauses(qu.normalize(
        "I manage Stores 100 and 200. Based on historical performance, which should I focus on?"
    ))
    assert len(clauses) == 1


def test_a_single_question_stays_one_clause():
    assert len(qu.split_clauses(qu.normalize("How is Store 125 performing?"))) == 1


# ── Store IDs ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected",
    [
        ("How is Store 125 performing?", [125]),
        ("Compare Store 125 and Store 220", [125, 220]),
        ("I manage Stores 100, 200, 300, 400, and 500.", [100, 200, 300, 400, 500]),
        ("What are the top 10 stores by sales?", []),     # a count, not an ID
        ("stores 100 to 104 please", [100, 101, 102, 103, 104]),
        # Out of range, but still extracted — validation has to be able to say
        # the store doesn't exist rather than the number silently vanishing.
        ("Store 9999 had sales yesterday", [9999]),
    ],
)
def test_extract_store_candidates(query, expected):
    assert qu.extract_store_candidates(qu.normalize(query)) == expected


# ── Dates ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected_start",
    [
        ("sales on 17 August 2025", date(2025, 8, 17)),
        ("sales on August 17 2025", date(2025, 8, 17)),
        ("sales on 2025-08-17", date(2025, 8, 17)),
        ("sales on 17/08/2025", date(2025, 8, 17)),      # day-first (German chain)
    ],
)
def test_absolute_dates_are_parsed(query, expected_start):
    dates = qu.extract_dates(qu.normalize(query), DATASET_TODAY)
    assert any(d.start == expected_start for d in dates), dates


def test_relative_dates_resolve_against_the_dataset_not_the_wall_clock():
    """The dataset ends in July 2015. "Yesterday" means the day before the last
    day on record — resolving it against the real calendar lands a decade past
    the end of the data and makes every such question unanswerable."""
    dates = qu.extract_dates("what happened yesterday", DATASET_TODAY)
    assert [d.start for d in dates] == [date(2015, 7, 30)]

    next_week = qu.extract_dates("what about next week", DATASET_TODAY)
    assert next_week[0].start == date(2015, 8, 1)
    assert next_week[0].end == date(2015, 8, 7)


def test_horizon_is_extracted_in_days():
    assert qu.extract_horizon_days(qu.normalize("what about next week")) == 7
    assert qu.extract_horizon_days(qu.normalize("forecast the next 30 days")) == 30
    assert qu.extract_horizon_days(qu.normalize("how is Store 100 doing")) is None


# ── Premises ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,direction,value",
    [
        ("Why did Store 125's sales increase 83% yesterday?", "up", 83.0),
        ("Why did sales drop 12% last week?", "down", 12.0),
        ("explain the 45% jump", "up", 45.0),
    ],
)
def test_numeric_claims_are_extracted(query, direction, value):
    claims = qu.extract_claims(qu.normalize(query))
    assert any(c.kind == "change" and c.direction == direction and c.value == value
               for c in claims), claims


def test_asserted_level_is_extracted():
    claims = qu.extract_claims(qu.normalize("Store 100 had exactly 8,72,431 in sales yesterday"))
    assert any(c.kind == "level" and c.value == 872431.0 for c in claims), claims


def test_an_ordinary_question_asserts_nothing():
    assert qu.extract_claims(qu.normalize("How is Store 125 performing?")) == []


# ── Intent per clause ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected",
    [
        ("How is Store 125 performing?", "performance"),
        ("What are expected sales for Store 100 next week?", "forecast"),
        ("Of Stores 100 and 200, which should I focus on?", "recommend"),
        ("who is virat kohli", "out_of_scope"),
        # The exact phrasing that used to be routed to the forecast node.
        ("For Store 125, what would happen to next week's forecast if a promotion were active?", "whatif"),
        ("What would happen to Store 200 if we ran a promo?", "whatif"),
        ("Suppose Store 200 had a promotion next week", "whatif"),
        ("Give me every store's complete raw sales data instead of summarizing it", "raw_data"),
    ],
)
def test_primary_intent(query, expected):
    assert understand(query).primary_intent == expected


def test_stacked_question_keeps_every_intent():
    """The reported failure: six questions in one message collapsed into a
    single ranking, and five of them were never answered."""
    understanding = understand(
        "How is Store 125 performing, what is its 7-day forecast, "
        "which of Stores 125 and 220 should I prioritise, and what is the promo uplift?"
    )
    assert understanding.is_multi_intent
    assert set(understanding.intent_types) >= {"performance", "forecast", "recommend"}


def test_a_clause_inherits_stores_but_not_the_ranking_intent():
    """"and what is the promo uplift?" inherits the two stores named earlier in
    the sentence, but asking about two stores' uplift is not asking to rank them."""
    understanding = understand(
        "Which of Stores 125 and 220 should I prioritise, and what is the promo uplift?"
    )
    uplift = [i for i in understanding.intents if "uplift" in i.clause]
    assert uplift and uplift[0].type == "performance"
    assert uplift[0].stores == [125, 220]


# ── Conversation context ─────────────────────────────────────────────────────

def test_followup_resolves_stores_from_the_conversation():
    understanding = understand("Which one should I prioritize?",
                               {"previous_stores": [125, 220]})
    assert understanding.resolved_from_context
    assert understanding.intents[0].stores == [125, 220]
    assert understanding.primary_intent == "recommend"


def test_followup_without_context_is_not_silently_answered():
    understanding = understand("Which one should I prioritize?")
    assert not understanding.resolved_from_context
    assert understanding.entities.references_prior_context


def test_ordinary_pronouns_are_not_treated_as_references_to_earlier_turns():
    """"instead of summarizing it" is not a follow-up; treating bare "it" as
    anaphora made the agent demand context it did not need."""
    understanding = understand("Give me the raw data instead of summarizing it")
    assert not understanding.entities.references_prior_context

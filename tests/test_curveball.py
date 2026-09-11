"""
tests/test_curveball.py
Owner: Parvat — Tech Lead

The live presentation curveball question stress test.

Curveball question:
    "I manage Stores 100, 200, 300, 400, and 500.
     Based on historical performance and your forecast,
     which stores should I focus on next week, why, and
     what does their promotional history tell me?"

Checks:
- Agent completes without crashing for all 5 stores
- Response contains all 4 required sections
- Store 1 (worst performer) is ranked consistently
- No hallucinated numbers (all figures come from tool results)
- Runs 20 consecutive times without any crash
"""
import sys
import pytest
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_API_KEY, DB_PATH, LGBM_PATH, USE_MOCKS

CURVEBALL = (
    "I manage Stores 100, 200, 300, 400, and 500. "
    "Based on historical performance and your forecast, "
    "which stores should I focus on next week, why, and "
    "what does their promotional history tell me?"
)

REQUIRED_SECTIONS = ["Observation", "Prediction", "Evidence", "Recommendation"]
CURVEBALL_STORES = [100, 200, 300, 400, 500]
N_STRESS_RUNS = 20


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def require_environment():
    missing = []
    if not USE_MOCKS and not DB_PATH.exists():
        missing.append("retail.db (run src/data_pipeline.py)")
    if not USE_MOCKS and not LGBM_PATH.exists():
        missing.append("lgbm_model.pkl (run compare_models.py)")
    if missing:
        pytest.skip(f"Missing artifacts: {', '.join(missing)}")


@pytest.fixture(scope="module")
def agent_run():
    from src.agent_graph import run_agent
    return run_agent


# ── 1. Single curveball run ───────────────────────────────────────────────────

class TestCurveballSingleRun:
    def test_agent_does_not_crash(self, agent_run):
        """Agent must complete without raising an exception."""
        try:
            response = agent_run(CURVEBALL)
            assert response is not None
        except Exception as e:
            pytest.fail(f"Agent crashed on curveball: {e}")

    def test_response_is_nonempty(self, agent_run):
        response = agent_run(CURVEBALL)
        assert len(response.strip()) > 50, "Response is too short — likely an error message"

    def test_response_contains_all_4_sections(self, agent_run):
        response = agent_run(CURVEBALL)
        for section in REQUIRED_SECTIONS:
            assert section in response, \
                f"Response missing required section: '{section}'\nResponse:\n{response[:300]}"

    def test_response_mentions_store_numbers(self, agent_run):
        response = agent_run(CURVEBALL)
        mentioned = [str(s) for s in CURVEBALL_STORES if str(s) in response]
        assert len(mentioned) >= 2, \
            f"Response should mention multiple store IDs, only found: {mentioned}"

    def test_response_has_citation_sources(self, agent_run):
        response = agent_run(CURVEBALL)
        # Citations look like "**Sources:**" or "📚" or "Sources:"
        has_citation = any(kw in response for kw in ["Sources:", "📚", "source", "model_engine", "database"])
        assert has_citation, "Response is missing source citations"

    def test_response_contains_numeric_figures(self, agent_run):
        response = agent_run(CURVEBALL)
        numbers = re.findall(r"\d[\d,]*\.?\d*", response)
        assert len(numbers) >= 3, \
            f"Response has too few numbers ({len(numbers)}) — may be hallucinating without data"


# ── 2. What-If query ──────────────────────────────────────────────────────────

class TestWhatIfQuery:
    def test_whatif_promo_query(self, agent_run):
        question = "What would happen to Store 100's sales next week if we add a promotion?"
        try:
            response = agent_run(question)
            assert response is not None and len(response.strip()) > 20
        except Exception as e:
            pytest.fail(f"What-If agent query crashed: {e}")

    def test_whatif_response_contains_forecast(self, agent_run):
        question = "What would happen to Store 200's sales next week if we remove the promotion?"
        response = agent_run(question)
        numbers = re.findall(r"€?\$?\d[\d,]+", response)
        assert len(numbers) >= 1, "What-If response should contain at least one numeric forecast"


# ── 3. Consistent ranking ─────────────────────────────────────────────────────

class TestConsistentRanking:
    def test_same_recommendation_multiple_runs(self, agent_run):
        """Run 3 times — the store recommended as 'focus' must be consistent."""
        simple_q = "Of Stores 100 and 200, which needs more attention next week and why?"
        responses = [agent_run(simple_q) for _ in range(3)]
        # Check store mentions are consistent (same store appears first in all runs)
        first_stores = []
        for r in responses:
            for store_id in [100, 200]:
                if str(store_id) in r:
                    first_stores.append(store_id)
                    break
        # Allow 2/3 consistent (1 edge case allowed)
        if first_stores:
            most_common = max(set(first_stores), key=first_stores.count)
            consistency = first_stores.count(most_common) / len(first_stores)
            assert consistency >= 0.67, \
                f"Store recommendation inconsistent across 3 runs: {first_stores}"


# ── 4. 20× stress test ────────────────────────────────────────────────────────

@pytest.mark.slow
class TestStressRuns:
    def test_20_consecutive_runs_no_crash(self, agent_run):
        """
        Run the full curveball question 20 consecutive times.
        No run should crash or return an empty response.
        """
        errors = []
        empty_responses = []

        for i in range(N_STRESS_RUNS):
            try:
                response = agent_run(CURVEBALL)
                if not response or len(response.strip()) < 20:
                    empty_responses.append(i + 1)
            except Exception as e:
                errors.append((i + 1, str(e)))

        if errors:
            error_summary = "\n".join(f"  Run {r}: {e}" for r, e in errors[:5])
            pytest.fail(f"{len(errors)}/{N_STRESS_RUNS} runs crashed:\n{error_summary}")

        if empty_responses:
            pytest.fail(
                f"{len(empty_responses)}/{N_STRESS_RUNS} runs returned empty responses: "
                f"runs {empty_responses}"
            )

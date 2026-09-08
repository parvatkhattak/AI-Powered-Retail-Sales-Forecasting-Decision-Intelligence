# Agent Design — LangGraph Architecture & Prompt Engineering

> Owner: Saumya — AI & Agent Engineer
> Covers: `src/agent_graph.py`, `src/decision_engine.py`, `src/prompts.py`

This document explains how the AI Assistant actually works end-to-end —
what the graph does, why the Decision Engine never calls an LLM, and how
hallucinations are prevented. See [`docs/architecture.md`](architecture.md)
for the whole-system view; this is the zoomed-in version of section 6.

---

## 1. Why an agent, not a single LLM call

A store manager's question ("which store should I focus on?") can't be
answered from an LLM's training data — it needs *this week's* real sales,
*this store's* real forecast, and *this store's* real promo history. Piping
a question straight to an LLM risks it inventing plausible-sounding numbers
(hallucination), which is unacceptable for a business recommendation.

So the agent's job is to **fetch real data first, then let the LLM only
format it** — never invent it. Concretely:

1. Classify what the question is asking (Router)
2. Call the real function(s) that answer it (DataAnalyst / Forecast / Decision)
3. Hand only that real data to the LLM, with an explicit instruction to
   never state a number that isn't in it (Respond)

## 2. Graph shape

```
        ┌──────────┐
 START →│  Router  │  extract store IDs (regex) + classify intent
        └────┬─────┘
             │  (routes to exactly one branch)
   ┌─────────┼──────────────┐
   ▼         ▼               ▼
DataAnalyst  Forecast      Decision
(database.py)(model_engine.py)(decision_engine.py, which itself
                                calls database.py + model_engine.py)
   │         │               │
   └─────────┴───────┬───────┘
                      ▼
                   Respond   (LLM formats the real data into prose,
                              or a deterministic template if no LLM
                              is configured)
                      │
                     END
```

This matches `docs/architecture.md` §6.2 — the Router dispatches to exactly
one downstream node per question, not a linear pipeline through all four.

### AgentState

```python
class AgentState(TypedDict, total=False):
    query: str                    # Original user question
    session_id: str               # For conversation memory
    intent: Intent                # "performance" | "forecast" | "recommend" | "whatif"
    store_ids: list[int]          # Extracted store IDs from query
    tool_results: dict            # Raw data returned by DataAnalyst/Forecast
    decision_report: dict         # Structured output from decision_engine
    response: str                 # Final formatted response for the user
    data_sources: list[str]       # Citation list for UI citation cards
    error: str | None             # Error message if something fails
```

## 3. The Router — intent classification

Two layers, so the agent works with or without an LLM configured:

1. **Store ID extraction** is always regex-based, never the LLM — a number
   only counts as a store ID if it follows the word "store"/"stores" in the
   question (`stores?\s*(?:id[s]?)?\s*[:#]?\s*((?:(?:\s*(?:,|and|&|/)\s*)*\d+\s*)+)`).
   This avoids false positives like "top 10 stores" or "next 7 days", and
   means we never hardcode a store ID anywhere in the codebase.
2. **Intent classification** first tries the LLM with structured output
   (`RouterOutput`, a Pydantic model constraining the answer to one of 4
   labels). If no `OPENROUTER_API_KEY` is configured, or the call fails for
   any reason, it falls back to a keyword classifier
   (`_classify_intent_fallback`) — so the agent is fully testable offline
   and never crashes for lack of a key.

## 4. Why `decision_engine.py` never calls an LLM

This is the most important design decision in this track. If store ranking
were left to the LLM, the same question could return a different #1 store
on different runs — unacceptable for a live demo, and a real risk noted in
`docs/architecture.md` §6.4 ("Random responses on repeated queries").

Instead, `generate_decision_report()` and `compare_stores_report()` compute
a **risk score from three real, measurable signals**:

| Signal | Source | Weight |
|---|---|---|
| Sales trend (declining?) | `database.get_store_metrics()`, first-half vs second-half average | up to 40 |
| Forecast weakness (below own average?) | `model_engine.get_7day_forecast()` vs 30-day average | up to 40 |
| Unused promo opportunity (>14 days since last promo) | `database.get_store_metrics()` promo flags | +20 |

Same inputs always produce the same score, so the same store always ranks
#1 for the same underlying data — verified by
`tests/test_decision_engine.py::test_ranking_is_deterministic_across_repeated_runs`
and `tests/test_agent_graph.py::test_curveball_consistent_top_store_across_20_runs`
(20 repeated runs, one identical top pick).

The LLM's only involvement is in the **Respond** node, which turns the
already-decided report into readable prose — it cannot change the ranking.

## 5. Hallucination prevention

Every system prompt in `prompts.py` ends with the same rule:

> "You may ONLY state sales figures, store IDs, percentages, and dates that
> appear in the tool results provided to you. If a figure is not in the
> tool results, say 'data not available' — never estimate or guess. Always
> end your answer with a line starting with '📚 Sources:' ..."

This is reinforced two ways beyond just the prompt text:

- The LLM is only ever shown the **already-computed** `tool_results` /
  `decision_report` dict as its only source of numbers — never asked to
  recall anything from its own training.
- `respond_node()` appends a `📚 Sources:` line in code if the LLM's
  response doesn't already include one, so citations are never silently
  dropped even if the LLM forgets the instruction.

`tests/test_decision_engine.py::test_numbers_in_report_trace_back_to_real_inputs`
checks this at the data layer too: every number quoted in a report's text
must equal a number the module itself computed, not a free-floating string.

## 6. Graceful degradation (no API key, no database yet)

Two failure modes are handled explicitly, both covered by tests:

- **No `OPENROUTER_API_KEY`** → router and respond nodes fall back to
  deterministic, non-LLM paths (keyword classification;
  template-formatted responses built directly from `tool_results` /
  `decision_report`). The agent still answers correctly, just without
  LLM-polished prose.
- **`USE_MOCKS=False` but `retail.db` / model files don't exist yet**
  (e.g. before Himanshu/Ashutosh's pipelines have been run locally) →
  `run_agent()` / `run_agent_stream()` catch the exception and return a
  `⚠️` prefixed friendly message instead of crashing the Streamlit app,
  per `docs/architecture.md` §13 ("Agent Error Handling").

## 7. What's intentionally out of scope here

- **What-If Node** — `docs/architecture.md` marks this "(S-Grade)". Until
  it exists as its own node, a `whatif`-classified question is routed to
  the Decision node, which still grounds its answer in a real forecast
  (`_route_from_intent` in `agent_graph.py`).
- **Real token-by-token LLM streaming** — `run_agent_stream()` currently
  streams the already-composed response word-by-word rather than
  streaming LLM tokens live. Swapping in the LLM's own async streaming
  is a small change confined to `run_agent_stream()`; the public contract
  (`Generator[str, None, None]`) doesn't need to change.

## 8. Test coverage

| File | Checks |
|---|---|
| `tests/test_decision_engine.py` | Required fields present, risk level valid, citations non-empty, ranking sorted correctly, ranking deterministic across 20 runs, unknown store degrades gracefully, quoted numbers trace back to real computed values |
| `tests/test_agent_graph.py` | Store ID extraction, intent classification (incl. the curveball question), no crashes across all 4 intents, citations present, multiple stores mentioned in a comparison, top pick stable across 20 runs, streaming output matches non-streaming, missing-DB failure is friendly not fatal, graph compiles |

Run with:
```bash
pytest tests/test_agent_graph.py tests/test_decision_engine.py -v
```
Both files run fully offline (no `OPENROUTER_API_KEY` required) since they
exercise the deterministic fallback paths described in section 6.

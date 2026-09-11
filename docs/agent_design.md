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

## 5b. Scope guardrail — refusing what the data can't answer

Hallucination prevention above keeps the LLM from inventing *numbers*. This
keeps the agent from answering questions it has no business answering at all.

The Router classifies an off-topic question as `out_of_scope`, which routes to
`out_of_scope_node` — a node that returns a fixed refusal **without touching
the database or the LLM**, then goes straight to `END`:

```
router --out_of_scope--> out_of_scope --> END     (no DB call, no LLM call)
```

A question is in scope if it names a store ID, or uses vocabulary from this
dataset (sales, forecast, promo, uplift, fleet, anomaly…). Everything else —
general knowledge, questions about the assistant, "ignore your instructions"
style prompts — gets the refusal.

Two details worth knowing:

- **The scope check is enforced in code, not delegated to the model.** An LLM
  asked to pick from a fixed set of labels will always pick one, so an
  off-topic question would otherwise be forced into `performance` and routed
  to a data node. `out_of_scope` exists in the router prompt too, but code has
  the final say.
- **Why it matters:** before this existed, *any* question with no store ID
  fell through to the data analyst node, which called `get_eda_summary()` — so
  "who is \<a cricketer\>" was answered with the chain's average daily revenue.
  `tests/test_agent_graph.py::test_off_topic_questions_get_a_refusal_not_sales_data`
  asserts the database is never even reached for those queries, and
  `test_in_scope_fleet_question_still_reaches_the_data_layer` guards the
  opposite failure — a legitimate fleet-wide question being refused.

## 5c. Guardrails enforced in code (`src/guardrails.py`)

A system prompt is a request, not a control — a model can be talked out of one.
So anything that actually matters is decided in code, in `guardrail_node`, the
**first node in the graph**. A blocked message never reaches the router, the
database, or the model:

```
START -> guardrail --blocked--> END        (refusal produced entirely in code)
             |
          allowed
             v
          router -> ... -> respond -> END
```

Screened categories, each with its own refusal so the user learns what was
actually refused: `prompt_injection`, `system_prompt`, `secrets`,
`destructive_db`, `raw_sql`, `internals`, `bulk_export`, plus empty and
over-length input. Patterns are word-boundary anchored so ordinary retail
phrasing survives — "sales **dropped** last month" is not `DROP TABLE`,
"**update** me on Store 100" is not a SQL `UPDATE`, and
`tests/test_guardrails.py` asserts both directions (30 adversarial prompts
blocked, 10 legitimate ones allowed).

On the way out, `redact_output()` strips anything credential-shaped from the
response, since the LLM's wording is not fully predictable.

### Why the architecture matters more than the patterns

Regex screening is the outer layer, not the real control. The reason
"run DROP TABLE" cannot work here is structural:

| Control | Status |
|---|---|
| Does the LLM get tools / function-calling? | **No** — `bind_tools` is never called |
| Can the LLM emit SQL that gets executed? | **No** — it only ever receives an already-computed dict to phrase |
| Is there any path executing caller-supplied SQL? | **No** — every query is a fixed literal in `database.py` with bound parameters |
| Database access mode | Read-only query functions; no INSERT/UPDATE/DELETE exists anywhere in the codebase |

So even if the screening were bypassed entirely, there is nothing for a
destructive instruction to reach. The screening exists so those attempts are
refused clearly and logged, rather than being answered with something
unhelpful.

## 6. Graceful degradation (no API key, no database yet)

Two failure modes are handled explicitly, both covered by tests:

- **No `OPENROUTER_API_KEY`** → router and respond nodes fall back to
  deterministic, non-LLM paths (keyword classification;
  template-formatted responses built directly from `tool_results` /
  `decision_report`). The agent still answers correctly, just without
  LLM-polished prose.

  > ⚠️ Because that fallback produces a perfectly plausible answer, a broken
  > LLM used to be **invisible** — the app looked like it was working when it
  > had never once reached OpenRouter. Both fallbacks now log a warning and
  > record the reason in `state["error"]`. To check directly, run:
  > ```python
  > from src.agent_graph import check_llm_connection
  > print(check_llm_connection())   # {"ok": True/False, "model": ..., "detail": ...}
  > ```
  > If `ok` is False, the response text you're seeing is the template, not the LLM.
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

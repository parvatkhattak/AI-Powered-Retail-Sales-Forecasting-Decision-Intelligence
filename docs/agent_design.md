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
START
  │
  ▼
Guardrail ──blocked──▶ END              (src/guardrails.py, refusal from code)
  │ allowed
  ▼
Understand                              (src/query_understanding.py)
  │   normalize → split into clauses → per-clause intent → entities
  │   → resolve anaphora against this session's earlier turns
  ▼
Validate                                (src/validation.py)
  │   store exists? date covered? horizon reachable? operation allowed?
  │   premise true? → build the executable plan
  ├──nothing answerable──▶ Refuse ──▶ END
  │ plan has steps
  ▼
Execute        one step per thing the user asked, in the order asked
  │   performance → database.py
  │   forecast    → model_engine.py
  │   whatif      → model_engine.get_whatif_forecast()
  │   recommend   → decision_engine.py
  ▼
Respond                                 (src/response_validation.py)
      compose (LLM or deterministic template) → run 10 checks on the
      composed text → fall back to the grounded template if it fails
  │
  ▼
 END
```

Each stage narrows what the answer is allowed to say. The LLM sits at the very
end, phrasing data that has already been fetched, bounded and checked.

### AgentState

```python
class AgentState(TypedDict, total=False):
    query: str                    # Original user question
    session_id: str               # For conversation memory
    intent: Intent                # Primary intent (first plan step)
    store_ids: list[int]          # Validated store IDs from the query
    understanding: object         # query_understanding.Understanding
    validation: object            # validation.ValidationResult
    steps: list[dict]             # One entry per executed plan step
    tool_results: dict            # Merged raw data returned by tool calls
    decision_report: dict         # Structured output from decision_engine
    response: str                 # Final formatted response for the user
    data_sources: list[str]       # Citation list for UI citation cards
    grounding: object             # Values the answer was allowed to state
    error: str | None             # Internal detail for logs — never shown
    blocked_reason: str | None    # Guardrail category when refused
    previous_stores: list[int]    # ─┐
    previous_intents: list[str]   #  │ conversation memory, carried
    previous_metrics: list[str]   #  │ between turns of one session
    previous_comparison: bool     #  │
    previous_date_range: list[str]# ─┘
```

## 3. Understand — raw text to structure

Before this layer existed the agent went straight from a string to an intent
label, so everything the question actually said was invisible: a question
about 2025 looked identical to one about last week, and a question about Store
9999 looked identical to one about Store 125.

`src/query_understanding.py` is pure, deterministic, and touches nothing:

1. **Normalize** — Unicode, dashes, quotes and digit grouping unified;
   contractions and shorthand expanded ("st 125 perf" → "store 125
   performance"); one canonical lowercase form for every matcher downstream.
   Apostrophes are required where the bare spelling is a real word, because
   expanding "its" to "it is" mangled "what is its forecast?".
2. **Split into clauses** — a stacked question is several questions in one
   string. Clauses that don't open like a question ("based on historical
   performance and your forecast") are merged into the question they qualify.
3. **Per-clause intent** — each clause gets its own label, so a six-part
   question stays six parts. The "more than one store means ranking"
   heuristic counts stores named *in that clause*, not ones inherited from
   the rest of the sentence.
4. **Entities** — store IDs (including out-of-range ones, so validation can
   say the store doesn't exist rather than the number silently vanishing),
   dates, forecast horizon, metrics, comparison, and any numeric claim the
   user asserted.
5. **Anaphora** — "which one should I prioritise?" resolves against
   `previous_stores` for the session.

**Store ID extraction is always regex-based, never the LLM** — a number only
counts as a store ID if it follows the word "store"/"stores", which avoids
false positives like "top 10 stores" or "next 7 days", and means no store ID
is ever hardcoded in the codebase.

### Dates resolve against the dataset, not the wall clock

Rossmann's history ends 2015-07-31. "Yesterday" can only sensibly mean the day
before the last day on record; resolving it against the real calendar lands a
decade past the end of the data and makes every relative question
unanswerable. `_reference_date()` returns the dataset's own latest date, read
from `database.get_dataset_bounds()`.

## 3b. Validate — what the data can actually support

`src/validation.py` checks the structure against real coverage, read from the
data rather than hardcoded:

| Check | Question it answers | Example refusal |
|---|---|---|
| `check_stores` | Is this a real store here? | "Store 9999 is not present in the available dataset." |
| `check_dates` | Is this date in history or the forecast window? | "2025-08-17 is outside what I can answer for." |
| `check_horizon` | Can the model see that far? | "You asked about the next 30 days, the model produces 7." |
| `check_operation` | Do we do this at all? | raw record dumps are refused, never substituted with a summary |
| `check_reference_resolution` | Does this follow-up point at anything? | "Your question refers back to something we haven't discussed yet." |
| `verify_claims` | Is the user's asserted number true? | "I can't verify an 83% increase. Store 125 recorded €13,146 … down 1.3%." |

A **blocking** finding removes the part of the question it applies to — not
necessarily the whole question. A stacked query that asks four answerable
things and two unanswerable ones answers the four and states the two
refusals; only when nothing executable is left does `refuse_node` produce the
whole answer.

The important property is that a refused question is never answered with a
*different* question's data. That was the shape of most of the reported
failures: the store didn't exist, or the date wasn't covered, or the
operation wasn't supported, and the user got a fleet-wide performance summary
that read like a confident answer.

## 3c. Intent classification and the LLM

Intent is decided in code, per clause — an LLM asked to pick from a fixed label
set will always pick something, so scope and routing cannot be delegated to it.
The agent makes **exactly one LLM call per question**, in the Respond node. The
old router round-trip was removed with the router node itself.

## 3d. Latency

The LLM dominates wall-clock time, so the code's job is to give it as little to
do as possible and to show the user what is happening meanwhile.

| Lever | Effect |
|---|---|
| `_llm_payload()` sends computed facts, not raw rows | single store 2,516 → 364 input tokens; five-store question 13,139 → 2,749 |
| `LLM_MAX_TOKENS` (config, env-overridable) | generation was unbounded; budget scales per section via `_token_budget()`, and `_looks_truncated()` discards a reply cut off at the ceiling |
| Prompts ask for the answer, not a narration | one live answer had listed six months of totals in full |
| No `get_baseline_comparison()` in the chat path | a second model loaded from disk plus two recursive forecasts per store, never quoted |
| `_read_reference_csv()` caches test.csv / store.csv | a five-store report re-read the same 1.4MB file ten times |
| Fleet screening skips SHAP | it feeds the Evidence prose and never the score |

Measured on the real database: what-if 1.6s → 0.08s, forecast 1.27s → 0.73s,
fleet risk 8.8s → 4.5s, full test suite 123s → 79s.

`progress_reporting(callback)` is a context manager that routes the current
thread's stage messages to a listener. `pages/4_AI_Assistant.py` runs the agent
on a worker thread and repaints a live elapsed timer and the current stage
("Running the 7-day forecast · 2.4s"), then reports the total on completion.
The slow paths — fleet ranking and multi-store comparison — take an optional
`progress` callback so they can count through stores rather than freezing on
one message. A listener that raises is swallowed: progress reporting must never
be able to break an answer.

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

### The risk level and the advice must agree

A deterministic score is not enough on its own. The first version wrote the
recommendation in a separate `if` chain from the score, and only `HIGH`
produced real advice — so a MEDIUM store got "Store 200 is stable — no urgent
action needed this week", and `compare_stores_report()` then quoted that
sentence after the word "Priority:". The ranking and the recommendation
contradicted each other in one line.

Now every level maps through one table:

| Risk level | Urgency | `action_required` | Shape of the advice |
|---|---|---|---|
| HIGH | `act_now` | True | "Act this week on Store N: …" |
| MEDIUM | `monitor` | True | "Watch Store N this week — \<the driver\>. Nothing needs emergency action yet." |
| LOW | `none` | False | "No action needed for Store N — no weakening signals." |

`_risk_drivers()` names which signal pushed the score up, so the advice cites a
reason rather than asserting a level. The comparison summary is phrased from
the top store's own urgency: it only says "Priority:" when that store's risk
level says action is due.

`check_report_consistency()` is the invariant, exported rather than left in the
tests because the agent runs it on its own output too
(`_run_recommend` logs an error if a report ever fails it):

- risk level and urgency must map to each other
- a store needing action must not carry "no action needed" wording
- a LOW-risk store must not carry priority wording
- the ranking must be ordered by risk score
- the summary must not name a priority while saying nothing needs doing

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
- **A real store with no forecast** — 259 of the 1,115 stores aren't in the
  forecast calendar the model's window is built from. That is a coverage fact
  about the store, so the answer says which it is
  (`_explain_missing_forecast`) and offers the store's real trading average
  and promo history, rather than reporting "no data available" or implying the
  simulation was run.
- **`USE_MOCKS=False` but `retail.db` / model files don't exist yet**
  (e.g. before Himanshu/Ashutosh's pipelines have been run locally) →
  `run_agent()` / `run_agent_stream()` catch the exception and return a
  `⚠️` prefixed friendly message instead of crashing the Streamlit app,
  per `docs/architecture.md` §13 ("Agent Error Handling").

## 6b. Response validation — checking the text, not trusting it

`src/response_validation.py` runs ten checks on the composed answer before it
leaves. A system prompt asking the model not to invent numbers is a request;
this is the check.

| Check | What it catches |
|---|---|
| `not_empty` | an answer that isn't one |
| `no_internal_leakage` | tracebacks, SQL, file paths, credential shapes |
| `numeric_grounding` | any currency amount or percentage with no source in the tool results |
| `store_grounding` | a store ID the answer names but never looked up |
| `date_grounding` | a date stated as if data existed for it |
| `required_notices` | a validation finding the user has to see, missing from the text |
| `no_self_contradiction` | one store called both a priority and stable |
| `covers_all_intents` | a stacked question answered with only one of its parts |
| `has_sources` | citations dropped |
| — | *(citations themselves are rewritten from the call log in `_assemble()`, never taken from the model: asked to cite "the sources you were given", one run cited the JSON key `tool_results`)* |
| `premise_corrected` | a claim the data contradicts, repeated as fact |

The deterministic composer passes `numeric_grounding` **by construction**:
`_money()`, `_pct()` and `_count()` register each figure with the `Grounding`
object as they format it, so it can only print numbers it was given. The LLM
gets no such registration — it only ever sees the context dict — so any figure
it invents is ungrounded and the check catches it. When the LLM's answer fails
any check, the grounded template is used instead and the failure is logged.

## 7. What's intentionally out of scope here

- **Real token-by-token LLM streaming** — `run_agent_stream()` currently
  streams the already-composed response word-by-word rather than
  streaming LLM tokens live. Swapping in the LLM's own async streaming
  is a small change confined to `run_agent_stream()`; the public contract
  (`Generator[str, None, None]`) doesn't need to change.

## 8. Test coverage

| File | Checks |
|---|---|
| `tests/test_decision_engine.py` | Required fields present, risk level valid, citations non-empty, ranking sorted correctly, ranking deterministic across 20 runs, unknown store degrades gracefully, quoted numbers trace back to real computed values |
| `tests/test_agent_graph.py` | Store ID extraction, intent classification (incl. the curveball question), no crashes across all 4 intents, citations present, multiple stores mentioned in a comparison, top pick stable across 20 runs, streaming output matches non-streaming, missing-DB failure is friendly not fatal, fleet questions route to different reports, graph compiles |
| `tests/test_guardrails.py` | 30 adversarial prompts blocked and categorised, 10 legitimate ones allowed, blocked prompts never reach the database, nonsense input doesn't crash or leak, different questions give different answers |
| `tests/test_query_understanding.py` | Normalization, clause splitting, qualifier merging, store-ID extraction (incl. out-of-range), absolute and relative date parsing, dataset-relative "yesterday", horizon extraction, premise extraction, per-clause intent, context resolution |
| `tests/test_agent_behaviour.py` | The ten reported failures, end to end: unknown store, date outside coverage, forecast horizon, false premise, what-if routing (and three paraphrases), conversational follow-up, six-part multi-intent, raw-data refusal (and three paraphrases), numeric grounding across six question shapes, decision consistency |

Run with:
```bash
pytest tests/test_agent_graph.py tests/test_decision_engine.py \
       tests/test_guardrails.py tests/test_query_understanding.py \
       tests/test_agent_behaviour.py -v
```
Both files run fully offline (no `OPENROUTER_API_KEY` required) since they
exercise the deterministic fallback paths described in section 6.

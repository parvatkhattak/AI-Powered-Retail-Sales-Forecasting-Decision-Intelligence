"""
src/agent_graph.py
Owner: Saumya — Agentic AI Engineer

Responsibilities:
- Define LangGraph state and nodes
- Understand, validate and route user queries
- Execute tool calls
- Format and check final responses

Graph design:

    START -> Guardrail -> Understand -> Validate -> {Execute | Refuse} -> Respond -> END

Each stage narrows what the answer is allowed to say:

    Guardrail    screens the message (src/guardrails.py)
    Understand   raw text -> structure: normalize, resolve context, extract
                 entities, detect every intent in the query
                 (src/query_understanding.py)
    Validate     check the structure against what the data supports — store
                 exists? date covered? horizon reachable? operation allowed?
                 premise true? — and build the executable plan
                 (src/validation.py)
    Execute      run the real tools for each plan step
    Respond      compose, then check the composed text against the grounded
                 data before it leaves (src/response_validation.py)

The LLM is never given tools, never sees the database, and never decides what
is answerable. It phrases data that has already been fetched and checked.
"""

import json
import logging
import re
import sys
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

from config import LLM_API_KEY, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from src import database, decision_engine, guardrails, model_engine, prompts
from src import query_understanding as qu
from src import response_validation as rv
from src import validation as dv

logger = logging.getLogger(__name__)

# Progress reporting. The UI shows what the agent is doing and how long it has
# been doing it, because a silent spinner for several seconds reads as a hang.
# Thread-local so a background worker reports only to its own listener.
_progress = threading.local()


@contextmanager
def progress_reporting(callback):
    """Route this thread's progress messages to `callback` for the duration."""
    previous = getattr(_progress, "callback", None)
    _progress.callback = callback
    try:
        yield
    finally:
        _progress.callback = previous


def _emit(stage: str) -> None:
    callback = getattr(_progress, "callback", None)
    if callback is None:
        return
    try:
        callback(stage)
    except Exception:  # a broken listener must never break the answer
        logger.debug("progress listener raised", exc_info=True)

Intent = Literal["performance", "forecast", "recommend", "whatif", "out_of_scope"]

_MAX_STORES_PER_QUERY = qu.MAX_STORES_PER_QUERY
_MIN_STORE_ID, _MAX_STORE_ID = 1, 1115


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
    grounding: object             # response_validation.Grounding used to check the answer
    error: str | None             # Internal detail for logs — never shown to the user
    blocked_reason: str | None    # Guardrail category when a message was refused
    safe_query: str | None        # The answerable part of a partly-refused message
    guardrail_notice: str | None  # Refusal that must be shown alongside the answer
    # Conversation memory, carried between turns of one session.
    previous_stores: list[int]
    previous_intents: list[str]
    previous_metrics: list[str]
    previous_comparison: bool
    previous_date_range: list[str]


# ── Backwards-compatible helpers ─────────────────────────────────────────────
# The understanding layer owns this logic now; these keep the public names other
# modules, docs and tests already use pointing at one implementation.

def _extract_store_ids(query: str) -> list[int]:
    """Store IDs in the query, bounded to the dataset's ID range."""
    return [
        sid for sid in qu.extract_store_candidates(query)
        if _MIN_STORE_ID <= sid <= _MAX_STORE_ID
    ][:_MAX_STORES_PER_QUERY]


def _mentions_more_stores_than_analysed(query: str) -> bool:
    return qu.mentions_more_stores_than(query)


def _parse_fleet_request(query: str) -> dict:
    return qu.parse_fleet_request(query)


def _is_in_scope(query: str, store_ids: list[int]) -> bool:
    return qu.is_in_scope(query, store_ids)


def _classify_intent_fallback(query: str, store_ids: list[int]) -> Intent:
    """Keyword-based classification for one query, with no LLM involved."""
    understanding = qu.understand(query, _reference_date())
    return understanding.primary_intent  # type: ignore[return-value]


# ── Conversation memory ──────────────────────────────────────────────────────

# What each session has already established. A follow-up like "which one should
# I prioritise?" names no store, so without this the agent had nothing to
# resolve the question against and fell back to a generic refusal — even though
# it had just been asked about two specific stores.
_SESSION_MEMORY: dict[str, dict] = {}
_MAX_SESSIONS_REMEMBERED = 200

_CONTEXT_KEYS = ("previous_stores", "previous_intents", "previous_metrics",
                 "previous_comparison", "previous_date_range")


def get_session_context(session_id: str) -> dict:
    return dict(_SESSION_MEMORY.get(session_id, {}))


def reset_session(session_id: str | None = None) -> None:
    """Clear one session's memory, or all of it."""
    if session_id is None:
        _SESSION_MEMORY.clear()
    else:
        _SESSION_MEMORY.pop(session_id, None)


def _remember(session_id: str, understanding, result) -> None:
    if len(_SESSION_MEMORY) > _MAX_SESSIONS_REMEMBERED:
        _SESSION_MEMORY.clear()

    # Accumulated, not replaced: three turns about Store 125 then Store 220
    # then "which one should I prioritise?" has to resolve to *both*, and
    # overwriting each turn left only the most recent one.
    earlier = _SESSION_MEMORY.get(session_id, {}).get("previous_stores", [])
    fresh = result.known_stores or list(understanding.entities.store_candidates)
    stores = list(dict.fromkeys(fresh + list(earlier)))[:_MAX_STORES_PER_QUERY]

    dates = understanding.entities.dates
    _SESSION_MEMORY[session_id] = {
        "previous_stores": sorted(stores),
        "previous_intents": understanding.intent_types,
        "previous_metrics": list(understanding.entities.metrics),
        "previous_comparison": understanding.entities.comparison,
        "previous_date_range": [dates[0].start.isoformat(), dates[-1].end.isoformat()] if dates else [],
    }


def _reference_date() -> date:
    """The dataset's own "today" — never the wall clock.

    Rossmann's history ends mid-2015, so resolving "yesterday" against the real
    date lands a decade past the end of the data. Every relative expression in
    a question resolves against the newest day on record instead.
    """
    return dv.get_coverage().history_end


# ── LLM ──────────────────────────────────────────────────────────────────────

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        if not LLM_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set — add it to your .env file.")
        _llm = ChatOpenRouter(model=LLM_MODEL, temperature=LLM_TEMPERATURE,
                              api_key=LLM_API_KEY, max_tokens=LLM_MAX_TOKENS)
    return _llm


def guardrail_node(state: AgentState) -> dict:
    """First node in the graph: screen the message before anything else runs.

    Runs ahead of everything so a blocked message never reaches the LLM, the
    database, or the model — the refusal is produced entirely from code.
    """
    _emit("Screening the request")
    verdict = guardrails.screen_input(state.get("query", ""))
    if verdict.blocked and verdict.safe_remainder:
        # Mixed message: the refused part is refused and said so, and the
        # legitimate question is still answered from approved aggregates.
        # Refusing the whole thing was safe but cost the user the half they
        # were entitled to.
        logger.info("guardrail refused part of a message (category=%s), answering the rest",
                    verdict.category)
        return {"blocked_reason": None, "safe_query": verdict.safe_remainder,
                "guardrail_notice": verdict.reply}
    if verdict.blocked:
        logger.info("guardrail blocked a message (category=%s)", verdict.category)
        return {"blocked_reason": verdict.category, "response": verdict.reply, "data_sources": []}
    return {"blocked_reason": None, "safe_query": None, "guardrail_notice": None}


def understand_node(state: AgentState) -> dict:
    """Raw text -> structure, with this session's earlier turns available."""
    # The answerable remainder when part of the message was refused, otherwise
    # the message itself.
    _emit("Reading the question")
    query = state.get("safe_query") or state["query"]
    context = get_session_context(state.get("session_id", "default"))
    coverage = dv.get_coverage()
    store_range = ((min(coverage.store_ids), max(coverage.store_ids))
                   if coverage.store_ids else (_MIN_STORE_ID, _MAX_STORE_ID))
    understanding = qu.understand(query, _reference_date(), context, store_id_range=store_range)

    return {
        "understanding": understanding,
        "store_ids": [s for s in understanding.entities.store_candidates
                      if _MIN_STORE_ID <= s <= _MAX_STORE_ID][:_MAX_STORES_PER_QUERY],
        "intent": understanding.primary_intent,
        **{key: context.get(key) for key in _CONTEXT_KEYS if context.get(key) is not None},
    }


def validate_node(state: AgentState) -> dict:
    """Check the structured question against what the data can support."""
    _emit("Checking what the data supports")
    understanding = state["understanding"]
    context = get_session_context(state.get("session_id", "default"))
    result = dv.validate(understanding, context)

    _remember(state.get("session_id", "default"), understanding, result)

    return {
        "validation": result,
        "store_ids": result.known_stores[:_MAX_STORES_PER_QUERY],
        "intent": result.plan[0].type if result.plan else understanding.primary_intent,
    }


# ── Tool handlers, one per intent ────────────────────────────────────────────

def _run_performance(step) -> dict:
    """Tools: get_store_metrics, get_promo_history, get_sales_trend (database.py),
    or the fleet-wide report that matches the question when no store is named."""
    tool_results: dict = {}
    sources: list[str] = []

    if step.stores:
        metrics_df = database.get_store_metrics(step.stores, days=30)
        tool_results["store_metrics"] = metrics_df.to_dict(orient="records")
        tool_results["promo_history"] = {sid: database.get_promo_history(sid) for sid in step.stores}
        tool_results["sales_trend"] = {
            sid: database.get_sales_trend(sid).to_dict(orient="records") for sid in step.stores
        }
        sources = ["database.get_store_metrics", "database.get_promo_history",
                   "database.get_sales_trend"]
    else:
        # No store named: pick the fleet-wide report that matches the question,
        # instead of always returning the same overall summary.
        request = qu.parse_fleet_request(step.clause)
        tool_results["fleet_request"] = request

        if request["kind"] == "promo_ranking":
            tool_results["promo_ranking"] = database.get_promo_uplift_ranking(
                top_n=request["n"]).to_dict(orient="records")
            sources = ["database.get_promo_uplift_ranking"]
        elif request["kind"] == "sales_ranking":
            tool_results["sales_ranking"] = database.get_store_sales_ranking(
                top_n=request["n"], ascending=request["ascending"]).to_dict(orient="records")
            sources = ["database.get_store_sales_ranking"]
        else:
            tool_results["eda_summary"] = database.get_eda_summary()
            sources = ["database.get_eda_summary"]

    return {"tool_results": tool_results, "sources": sources}


def _run_forecast(step) -> dict:
    """Tools: get_7day_forecast, get_shap_explanations (model_engine.py).

    Deliberately does *not* call get_baseline_comparison(): it loads a second
    model from disk and runs two more recursive forecasts (~0.4s per store),
    and the chat answer never quotes it — the baseline-vs-model chart lives on
    the Forecasting page, which calls it directly. It was pure latency here.
    """
    tool_results: dict = {"forecast": {}, "shap": {}, "no_forecast": {}}
    sources: list[str] = []

    tool_results["calendar"] = {}

    for sid in step.stores:
        forecast = model_engine.get_7day_forecast(sid)
        if forecast is None or forecast.empty:
            tool_results["no_forecast"][sid] = _explain_missing_forecast(sid)
            continue
        tool_results["calendar"][sid] = _forecast_calendar(sid)
        tool_results["forecast"][sid] = forecast.to_dict(orient="records")
        tool_results["shap"][sid] = model_engine.get_shap_explanations(sid)
        sources = ["model_engine.get_7day_forecast", "model_engine.get_shap_explanations"]

    return {"tool_results": tool_results, "sources": sources}


def _forecast_calendar(store_id: int) -> list[dict]:
    """Open/Promo/holiday flags for the forecast window, as known in advance.

    Needed because a closed day forecasts as exactly 0 — correctly, a shut shop
    sells nothing — and without the calendar the answer had no way to tell that
    apart from a predicted collapse. It was being read as underperformance.
    """
    try:
        calendar = model_engine.get_forecast_calendar(store_id)
    except Exception as exc:
        logger.warning("forecast calendar unavailable for store %s: %s", store_id, exc)
        return []
    if calendar is None or calendar.empty:
        return []
    return [
        {"date": str(row["Date"])[:10], "open": int(row["Open"]),
         "promo": int(row["Promo"]),
         "state_holiday": str(row["StateHoliday"]),
         "school_holiday": int(row["SchoolHoliday"])}
        for _, row in calendar.iterrows()
    ]


def _explain_missing_forecast(store_id: int) -> dict:
    """Why a real store has no forecast — said plainly instead of "no data".

    259 of the 1,115 stores aren't in the forecast calendar the model window is
    built from. That is a coverage fact about this store, not a failure, and the
    answer should say which it is.
    """
    detail = {"store_id": store_id, "reason": "not_in_forecast_window"}
    try:
        info = model_engine.get_missing_store_info(store_id)
    except Exception as exc:
        logger.warning("could not explain missing forecast for store %s: %s", store_id, exc)
        info = None

    if info:
        detail["recent_avg_sales"] = round(float(info["recent_avg"]), 2)
        detail["window_start"] = str(info["dates"][0])[:10]
        detail["window_end"] = str(info["dates"][-1])[:10]

    try:
        coverage = model_engine.get_forecast_coverage()
        detail["stores_with_forecast"] = coverage.get("stores_with_forecast")
        detail["total_stores"] = dv.get_coverage().total_stores or None
    except Exception as exc:
        logger.warning("forecast coverage unavailable: %s", exc)
    return detail


def _run_whatif(step) -> dict:
    """Tools: get_whatif_forecast with the promo forced on and off.

    This node is the fix for what-if questions being answered as plain
    forecasts: `model_engine.get_whatif_forecast()` existed the whole time and
    nothing in the agent ever called it, so "what would happen if a promotion
    were active" got whatever the ordinary forecast path produced.
    """
    tool_results: dict = {"whatif": {}, "no_forecast": {}}
    sources: list[str] = []

    for sid in step.stores:
        with_promo = model_engine.get_whatif_forecast(sid, True)
        without_promo = model_engine.get_whatif_forecast(sid, False)

        if with_promo is None or with_promo.empty or without_promo is None or without_promo.empty:
            tool_results["no_forecast"][sid] = _explain_missing_forecast(sid)
            # A store with no forecast coverage can still be answered with its
            # real promo history, which is what the question is actually about.
            try:
                tool_results.setdefault("promo_history", {})[sid] = database.get_promo_history(sid)
                sources.append("database.get_promo_history")
            except Exception as exc:
                logger.warning("promo history unavailable for store %s: %s", sid, exc)
            continue

        # Closed days forecast as 0 in both scenarios and would drag both daily
        # averages down by the same ~1/7, making the promotion look weaker than
        # it is and reading as predicted collapse rather than a shut shop.
        # The scenario is compared across trading days only.
        open_with = with_promo[with_promo["PredictedSales"] > 0]
        open_without = without_promo[without_promo["PredictedSales"] > 0]
        trading_days = max(len(open_with), len(open_without))
        closed_days = int(len(with_promo) - trading_days)

        with_avg = float(open_with["PredictedSales"].mean()) if not open_with.empty else 0.0
        without_avg = float(open_without["PredictedSales"].mean()) if not open_without.empty else 0.0
        with_total = float(with_promo["PredictedSales"].sum())
        without_total = float(without_promo["PredictedSales"].sum())

        tool_results["whatif"][sid] = {
            "scenario": "promotion active on every open day of the forecast window",
            "with_promo_avg": round(with_avg, 2),
            "without_promo_avg": round(without_avg, 2),
            "difference": round(with_avg - without_avg, 2),
            "difference_pct": round((with_avg - without_avg) / without_avg * 100, 2) if without_avg else 0.0,
            "with_promo_total": round(with_total, 2),
            "without_promo_total": round(without_total, 2),
            "total_difference": round(with_total - without_total, 2),
            "days": len(with_promo),
            "trading_days": trading_days,
            "closed_days": closed_days,
            "window_start": str(with_promo["Date"].iloc[0])[:10],
            "window_end": str(with_promo["Date"].iloc[-1])[:10],
            "with_promo_daily": with_promo.to_dict(orient="records"),
            "without_promo_daily": without_promo.to_dict(orient="records"),
        }
        sources.append("model_engine.get_whatif_forecast")
        # Fetched for covered stores too, so the interpretation can say whether
        # the modelled effect is bigger or smaller than what promotions have
        # actually achieved at this store before.
        try:
            tool_results.setdefault("promo_history", {})[sid] = database.get_promo_history(sid)
            sources.append("database.get_promo_history")
        except Exception as exc:
            logger.warning("promo history unavailable for store %s: %s", sid, exc)

    return {"tool_results": tool_results, "sources": sorted(set(sources))}


# A fleet-wide risk question shows this many stores unless it asks for another
# number, and screens this many candidates before scoring them.
_FLEET_RISK_DEFAULT_N = 5
_FLEET_RISK_MAX_N = 10


def _run_recommend(step) -> dict:
    """Calls decision_engine.py, which internally fetches from both database.py
    and model_engine.py."""
    if not step.stores and qu.is_fleet_risk_request(step.clause):
        # "Which stores are most at risk?" names no store, and used to fall
        # through to "no specific store was identified" even though the risk
        # methodology and the fleet data to run it on both already existed.
        requested = qu.parse_fleet_request(step.clause).get("n") or _FLEET_RISK_DEFAULT_N
        top_n = max(1, min(int(requested), _FLEET_RISK_MAX_N))
        report = decision_engine.rank_fleet_risk(top_n=top_n, progress=_emit)
        sources = sorted({s for r in report["ranked_stores"] for s in r["data_sources"]}
                         | {"database.get_fleet_trend_screen"})
        violations = decision_engine.check_report_consistency(report)
        if violations:
            logger.error("fleet risk report failed its consistency check: %s", violations)
        return {"tool_results": {"decision_report": report}, "sources": sources,
                "decision_report": report, "consistency_violations": violations}

    if len(step.stores) > 1:
        report = decision_engine.compare_stores_report(step.stores, progress=_emit)
        sources = sorted({s for r in report["ranked_stores"] for s in r["data_sources"]})
    elif len(step.stores) == 1:
        report = decision_engine.generate_decision_report(step.stores[0])
        sources = report["data_sources"]
    else:
        report = {"summary": "No specific store was identified in the question.", "ranked_stores": []}
        sources = []

    # A report that contradicts itself must never reach the user; this is the
    # same invariant the decision engine's own tests assert.
    violations = decision_engine.check_report_consistency(report)
    if violations:
        logger.error("decision report failed its consistency check: %s", violations)

    return {"tool_results": {"decision_report": report}, "sources": sources,
            "decision_report": report, "consistency_violations": violations}


_HANDLERS = {
    "performance": _run_performance,
    "forecast": _run_forecast,
    "recommend": _run_recommend,
    "whatif": _run_whatif,
}


def _STEP_PROGRESS_LABEL(spec) -> str:
    """What the user is told is happening while a plan step runs."""
    where = ("Store " + ", ".join(map(str, spec.stores))) if spec.stores else "the fleet"
    return {
        "performance": f"Fetching sales history for {where}",
        "forecast": f"Running the 7-day forecast for {where}",
        "whatif": f"Simulating the promotion for {where}",
        "recommend": (f"Scoring risk across the fleet" if not spec.stores
                      else f"Scoring risk for {where}"),
    }.get(spec.type, f"Working on {where}")


def execute_node(state: AgentState) -> dict:
    """Run every step of the validated plan, in the order the user asked."""
    result = state["validation"]
    steps: list[dict] = []
    merged: dict = {}
    sources: list[str] = []
    decision_report: dict = {}
    error = state.get("error")

    for spec in result.plan:
        handler = _HANDLERS.get(spec.type)
        if handler is None:
            continue
        _emit(_STEP_PROGRESS_LABEL(spec))
        try:
            outcome = handler(spec)
        except Exception as exc:
            # One failing step must not lose the answers to the others.
            logger.warning("plan step %s failed: %s", spec.type, exc)
            error = f"{error + ' | ' if error else ''}step_{spec.type}: {exc}"
            if len(result.plan) == 1:
                raise
            continue

        steps.append({
            "intent": spec.type,
            "stores": spec.stores,
            "scope": spec.scope,
            "metrics": spec.metrics,
            "clause": spec.clause,
            "tool_results": outcome["tool_results"],
            "sources": outcome["sources"],
        })
        merged.update(outcome["tool_results"])
        sources += outcome["sources"]
        if outcome.get("decision_report"):
            decision_report = outcome["decision_report"]

    return {
        "steps": steps,
        "tool_results": merged,
        "decision_report": decision_report,
        "data_sources": list(dict.fromkeys(sources)),
        "error": error,
    }


OUT_OF_SCOPE_REPLY = (
    "I'm the Retail AI assistant for this store network, so I can only answer "
    "questions that this project's sales database and forecasting model can "
    "actually back up — I won't guess at anything outside that.\n\n"
    "Things I can answer:\n"
    "- **Store performance** — \"How is Store 125 performing?\"\n"
    "- **7-day forecasts** — \"What are expected sales for Store 300 next week?\"\n"
    "- **Promotions** — \"Which stores have the highest promo uplift?\"\n"
    "- **Where to focus** — \"I manage Stores 100, 200 and 300 — which needs attention?\""
)


def refuse_node(state: AgentState) -> dict:
    """The answer when validation left nothing answerable.

    This is the node that stops the worst failure mode: a question the data
    can't support being answered with a different question's data. If the store
    doesn't exist, or the date is outside coverage, or the operation isn't one
    this assistant performs, that is the answer — not a fleet summary.
    """
    result = state.get("validation")
    findings = result.blocking_findings if result else []

    if not findings:
        return {"response": OUT_OF_SCOPE_REPLY, "data_sources": [], "tool_results": {}}

    parts = [f.message for f in findings]
    extra = [f.message for f in (result.notices if result else [])]
    if state.get("guardrail_notice"):
        parts.insert(0, state["guardrail_notice"])
    body = "\n\n".join(parts + extra)

    # Say what *can* be asked, so a refusal is still useful.
    body += (
        "\n\nWhat I can answer instead:\n"
        "- **Store performance** — \"How is Store 125 performing?\"\n"
        "- **7-day forecasts** — \"What are expected sales for Store 300 next week?\"\n"
        "- **Promotions** — \"Which stores have the highest promo uplift?\"\n"
        "- **Where to focus** — \"Of Stores 125 and 220, which needs attention?\""
    )
    return {"response": body, "data_sources": [], "tool_results": {}}


# ── Deterministic composition ────────────────────────────────────────────────
# Every figure printed here is registered with the Grounding object as it is
# formatted, so this path passes the numeric-grounding check by construction.

def _money(value, grounding: rv.Grounding) -> str:
    grounding.add_number(value)
    return f"€{value:,.0f}"


def _pct(value, grounding: rv.Grounding, digits: int = 1) -> str:
    grounding.add_number(value)
    return f"{value:.{digits}f}%"


def _count(value, grounding: rv.Grounding) -> str:
    grounding.add_number(value)
    return f"{value:,}"


def _performance_section(step: dict, grounding: rv.Grounding) -> str:
    tool_results = step["tool_results"]

    if step["stores"] and tool_results.get("store_metrics"):
        wants_only_promo = step["metrics"] == ["promo_uplift"]
        summaries = [
            _summarise_store_performance(sid, tool_results["store_metrics"],
                                         (tool_results.get("promo_history") or {}).get(sid),
                                         grounding, promo_only=wants_only_promo)
            for sid in step["stores"]
        ]
        return "\n\n".join(s for s in summaries if s)

    if tool_results.get("sales_ranking"):
        rows = tool_results["sales_ranking"]
        ascending = tool_results.get("fleet_request", {}).get("ascending", False)
        for row in rows:
            grounding.add_store(row["Store"])

        if len(rows) == 1:
            r = rows[0]
            superlative = "lowest-performing" if ascending else "best-performing"
            return (
                f"**Store {r['Store']} is the {superlative} store**, averaging "
                f"**{_money(r['avg_daily_sales'], grounding)}/day**.\n\n"
                f"That's {_money(r['total_sales'], grounding)} in total sales across "
                f"{_count(r['days_trading'], grounding)} trading days."
            )

        label = "Lowest" if ascending else "Top"
        grounding.add_number(len(rows))
        lines = [f"**{label} {len(rows)} stores by average daily sales**\n"]
        lines += [
            f"{i}. **Store {r['Store']}** — {_money(r['avg_daily_sales'], grounding)}/day "
            f"({_money(r['total_sales'], grounding)} total over "
            f"{_count(r['days_trading'], grounding)} trading days)"
            for i, r in enumerate(rows, start=1)
        ]
        return "\n".join(lines)

    if tool_results.get("promo_ranking"):
        rows = tool_results["promo_ranking"]
        for row in rows:
            grounding.add_store(row["Store"])

        if len(rows) == 1:
            r = rows[0]
            return (
                f"**Store {r['Store']} has the highest promotional uplift** at "
                f"**{_pct(r['uplift_pct'], grounding)}**.\n\n"
                f"It averages {_money(r['promo_avg_sales'], grounding)}/day on promo days versus "
                f"{_money(r['non_promo_avg_sales'], grounding)}/day without one."
            )

        grounding.add_number(len(rows))
        lines = [f"**Top {len(rows)} stores by promotional uplift**\n"]
        lines += [
            f"{i}. **Store {r['Store']}** — {_pct(r['uplift_pct'], grounding)} uplift "
            f"(promo {_money(r['promo_avg_sales'], grounding)}/day vs non-promo "
            f"{_money(r['non_promo_avg_sales'], grounding)}/day)"
            for i, r in enumerate(rows, start=1)
        ]
        return "\n".join(lines)

    if tool_results.get("eda_summary"):
        s = tool_results["eda_summary"]
        grounding.add_store(s.get("best_store_id"))
        grounding.add_store(s.get("worst_store_id"))
        return (
            f"**Average daily sales across the fleet: "
            f"{_money(s.get('avg_daily_sales', 0), grounding)}** "
            f"({_count(s.get('total_stores', 0), grounding)} stores).\n\n"
            f"- Best performer: **Store {s.get('best_store_id')}** at "
            f"{_money(s.get('best_store_avg_sales', 0), grounding)}/day\n"
            f"- Lowest performer: **Store {s.get('worst_store_id')}** at "
            f"{_money(s.get('worst_store_avg_sales', 0), grounding)}/day"
        )

    return ""


def _store_performance_facts(store_id: int, metric_rows: list[dict],
                             promo: dict | None) -> dict:
    """The handful of figures a performance answer is actually made of.

    One place computes them, and both consumers read from here: the
    deterministic renderer below, and the compact context handed to the LLM.
    Sending the raw daily rows instead was costing ~13k input tokens on a
    five-store question — most of the latency, for numbers no writer needs.
    """
    sales = [
        r["sales"] for r in metric_rows
        if r.get("store_id") == store_id and r.get("sales") is not None
    ]
    if not sales:
        return {}

    midpoint = len(sales) // 2
    earlier, recent = sales[:midpoint], sales[midpoint:]
    if midpoint and sum(earlier):
        trend = (sum(recent) / len(recent) - sum(earlier) / len(earlier)) / (sum(earlier) / len(earlier)) * 100
    else:
        trend = None

    facts = {
        "store_id": store_id,
        "days_counted": len(sales),
        "avg_daily_sales": round(sum(sales) / len(sales), 2),
        "trend_pct": round(trend, 2) if trend is not None else None,
        "best_day_sales": round(max(sales), 2),
        "quietest_day_sales": round(min(sales), 2),
    }
    if promo:
        facts.update({
            "promo_uplift_pct": promo.get("uplift_pct"),
            "promo_avg_sales": promo.get("promo_avg_sales"),
            "non_promo_avg_sales": promo.get("non_promo_avg_sales"),
        })
    return facts


def _summarise_store_performance(store_id: int, metric_rows: list[dict], promo: dict | None,
                                 grounding: rv.Grounding, promo_only: bool = False) -> str:
    """Plain-language performance summary for one store — average, direction of
    travel, promo uplift — built from rows already fetched."""
    grounding.add_store(store_id)
    facts = _store_performance_facts(store_id, metric_rows, promo)

    if promo_only:
        if not promo or not promo.get("uplift_pct"):
            return f"No promotional history is recorded for Store {store_id}."
        return (
            f"**Store {store_id} promotional uplift: {_pct(promo['uplift_pct'], grounding)}** — "
            f"{_money(promo.get('promo_avg_sales', 0), grounding)}/day on promo days versus "
            f"{_money(promo.get('non_promo_avg_sales', 0), grounding)}/day without one."
        )

    if not facts:
        return ""

    if facts["trend_pct"] is None:
        trend = "Not enough history yet to read a trend."
    else:
        direction = "up" if facts["trend_pct"] >= 0 else "down"
        trend = f"Sales are **{direction} {_pct(abs(facts['trend_pct']), grounding)}** across that period."

    lines = [
        f"**Store {store_id} is averaging {_money(facts['avg_daily_sales'], grounding)}/day** over the last "
        f"{_count(facts['days_counted'], grounding)} trading days.\n",
        f"- {trend}",
        f"- Best day {_money(facts['best_day_sales'], grounding)}, "
        f"quietest day {_money(facts['quietest_day_sales'], grounding)}",
    ]
    if promo and promo.get("uplift_pct"):
        lines.append(
            f"- Promotions lift this store **{_pct(promo['uplift_pct'], grounding)}** "
            f"({_money(promo.get('promo_avg_sales', 0), grounding)}/day on promo vs "
            f"{_money(promo.get('non_promo_avg_sales', 0), grounding)}/day without)"
        )
    return "\n".join(lines)


def _forecast_section(step: dict, grounding: rv.Grounding) -> str:
    tool_results = step["tool_results"]
    lines: list[str] = []

    calendars = tool_results.get("calendar") or {}
    for sid, rows in (tool_results.get("forecast") or {}).items():
        if not rows:
            continue
        grounding.add_store(sid)
        for row in rows:
            grounding.add_date(str(row.get("Date")))
        closed = _closed_days(rows, calendars.get(sid))
        trading = [r["PredictedSales"] for r in rows
                   if str(r.get("Date"))[:10] not in closed]
        if not trading:
            trading = [r["PredictedSales"] for r in rows]

        first_date, last_date = str(rows[0].get("Date"))[:10], str(rows[-1].get("Date"))[:10]
        lines.append(
            f"**Store {sid}: {_money(sum(trading) / len(trading), grounding)}/day forecast** across "
            f"its {_count(len(trading), grounding)} trading days, "
            f"{first_date} to {last_date} ({_money(sum(trading), grounding)} in total)."
        )
        if closed:
            listed = ", ".join(sorted(closed))
            lines.append(
                f"- {listed} {'are' if len(closed) > 1 else 'is'} a scheduled **closed day** for "
                f"this store, so the model forecasts €0 for "
                f"{'them' if len(closed) > 1 else 'it'}. That is a shut shop, not weak trading, "
                f"and it is excluded from the daily average above."
            )

    for sid, detail in (tool_results.get("no_forecast") or {}).items():
        grounding.add_store(sid)
        note = (
            f"**No forecast is available for Store {sid}.** It isn't in the forecast "
            f"calendar the model's window is built from, so I won't estimate a number for it."
        )
        if detail.get("recent_avg_sales"):
            note += (
                f" What I do have is its recent trading average: "
                f"{_money(detail['recent_avg_sales'], grounding)}/day."
            )
        lines.append(note)

    return "\n\n".join(lines)


def _closed_days(forecast_rows: list[dict], calendar: list[dict] | None) -> set[str]:
    """Dates in the forecast window on which this store is scheduled to be shut.

    Taken from the store's own forecast calendar where available. A zero
    prediction is only treated as a closed day when the calendar says the store
    is closed — a genuine zero prediction on an open day would be a real signal
    and must not be silently reclassified.
    """
    if calendar:
        return {day["date"] for day in calendar if not day.get("open")}
    return set()


def _whatif_section(step: dict, grounding: rv.Grounding) -> str:
    tool_results = step["tool_results"]
    lines: list[str] = []

    for sid, sim in (tool_results.get("whatif") or {}).items():
        grounding.add_store(sid)
        sign = "+" if sim["difference"] >= 0 else "-"
        window = f"{sim['window_start']} to {sim['window_end']}" if sim.get("window_start") else "next week"
        grounding.add_date(sim.get("window_start"))
        grounding.add_date(sim.get("window_end"))

        closed_note = ""
        if sim.get("closed_days"):
            closed_note = (
                f" ({_count(sim['closed_days'], grounding)} scheduled closed day"
                f"{'s' if sim['closed_days'] != 1 else ''} excluded — a shut store forecasts €0, "
                f"which is not weak trading)"
            )

        lines.append(
            f"**Store {sid} — what if a promotion ran every open day, {window}?**\n\n"
            f"**Baseline (promotion off)** — {_money(sim['without_promo_avg'], grounding)}/day "
            f"across {_count(sim['trading_days'], grounding)} trading days, "
            f"{_money(sim['without_promo_total'], grounding)} over the window{closed_note}\n\n"
            f"**Promotion on** — {_money(sim['with_promo_avg'], grounding)}/day, "
            f"{_money(sim['with_promo_total'], grounding)} over the same window\n\n"
            f"**Difference** — **{sign}{_money(abs(sim['difference']), grounding)}/day, "
            f"{sign}{_pct(abs(sim['difference_pct']), grounding)}** "
            f"({sign}{_money(abs(sim['total_difference']), grounding)} across the window)\n\n"
            f"**Interpretation** — {_whatif_interpretation(sid, sim, tool_results, grounding)}"
        )

    for sid, detail in (tool_results.get("no_forecast") or {}).items():
        grounding.add_store(sid)
        covered = detail.get("stores_with_forecast")
        total = detail.get("total_stores")
        scope = (f" The model's forecast window covers "
                 f"{_count(covered, grounding)} of the {_count(total, grounding)} stores in this "
                 f"dataset; Store {sid} is one of the "
                 f"{_count(total - covered, grounding)} it does not.") if covered and total else ""
        note = (
            f"**I can't simulate a promotion for Store {sid} — data not available.** The "
            f"scenario works by re-running the forecast with the promotion flag flipped, so it "
            f"needs a forecast to vary. Store {sid} has no forecast in this window, so there is "
            f"nothing to compare against and I won't manufacture one by applying its historical "
            f"uplift to a number the model never produced.{scope}"
        )
        promo = (tool_results.get("promo_history") or {}).get(sid)
        if promo and promo.get("uplift_pct"):
            note += (
                f"\n\nSeparately — and this is history, not a simulation — promotions have "
                f"lifted Store {sid} by **{_pct(promo['uplift_pct'], grounding)}** on average "
                f"across its recorded past: {_money(promo.get('promo_avg_sales', 0), grounding)}/day "
                f"on promo days versus {_money(promo.get('non_promo_avg_sales', 0), grounding)}/day "
                f"without. That is what the store has done before, not what the model predicts it "
                f"would do next week."
            )
        if detail.get("recent_avg_sales"):
            note += f" Its recent trading average is {_money(detail['recent_avg_sales'], grounding)}/day."
        lines.append(note)

    return "\n\n".join(lines)


def _whatif_interpretation(store_id: int, sim: dict, tool_results: dict,
                           grounding: rv.Grounding) -> str:
    """Whether the scenario looks worthwhile, from the modelled numbers alone.

    Kept strictly to what this dataset contains. Rossmann has no promotion
    cost, margin or stock data, so "worthwhile" can only ever be a statement
    about incremental revenue — saying anything about profit would be inventing
    the half of the calculation nobody gave us.
    """
    total_gain = sim["total_difference"]
    pct = sim["difference_pct"]
    promo = (tool_results.get("promo_history") or {}).get(store_id) or {}
    historical = promo.get("uplift_pct")

    if pct <= 0:
        verdict = (
            f"the model expects **no gain** from running the promotion in this particular week "
            f"— it forecasts {_money(abs(total_gain), grounding)} "
            f"{'less' if total_gain < 0 else 'no more'} revenue across the window."
        )
    else:
        verdict = (
            f"the model expects the promotion to add "
            f"**{_money(total_gain, grounding)}** across the window "
            f"({_pct(pct, grounding)} more revenue)."
        )

    context = ""
    if historical:
        gap = "below" if pct < historical else "above"
        context = (
            f" For context, this store's *historical* promo uplift is "
            f"{_pct(historical, grounding)}, so the modelled effect is {gap} what promotions have "
            f"achieved here in the past — the model is accounting for this specific week's "
            f"conditions, not repeating the long-run average."
        )

    caveat = (
        " Whether that is worth doing depends on the promotion's cost and margin, which this "
        "dataset does not contain — so this is an expected-revenue figure, not a profit one."
    )
    return verdict + context + caveat


def _recommend_section(step: dict, grounding: rv.Grounding) -> str:
    report = step["tool_results"].get("decision_report") or {}

    if report.get("ranked_stores"):
        if step["stores"]:
            header = f"**Priority ranking for stores {', '.join(str(s) for s in step['stores'])}:**"
        else:
            # Fleet-wide: say how the shortlist was reached, so a relative
            # ranking is never mistaken for an absolute alarm.
            count = len(report["ranked_stores"])
            grounding.add_number(count)
            header = f"**The {count} stores most at risk across the fleet:**"
        lines = [header + "\n"]
        for i, r in enumerate(report["ranked_stores"], start=1):
            grounding.add_store(r["store_id"])
            grounding.add_number(r["risk_score"])
            grounding.ingest(r)
            lines.append(
                f"{i}. **Store {r['store_id']}** — risk: {r['risk_level']} "
                f"(score {r['risk_score']})\n"
                f"   - 🔍 Observation: {r['observation']}\n"
                f"   - 📈 Prediction: {r['prediction']}\n"
                f"   - 📊 Evidence: {r['evidence']}\n"
                f"   - ✅ Recommendation: {r['recommendation']}"
            )
        lines.append(f"\n**Summary:** {report['summary']}")
        if report.get("methodology"):
            grounding.add_number(report.get("screened_stores"))
            grounding.add_number(report.get("fleet_size"))
            lines.append(f"\n*How this was ranked:* {report['methodology']}")
        return "\n".join(lines)

    if "observation" in report:
        grounding.ingest(report)
        return (
            f"🔍 **Observation:** {report['observation']}\n\n"
            f"📈 **Prediction:** {report['prediction']}\n\n"
            f"📊 **Evidence:** {report['evidence']}\n\n"
            f"✅ **Recommendation:** {report['recommendation']}"
        )

    return ""


_SECTION_BUILDERS = {
    "performance": _performance_section,
    "forecast": _forecast_section,
    "recommend": _recommend_section,
    "whatif": _whatif_section,
}

_SECTION_TITLES = {
    "performance": "Recent performance",
    "forecast": "7-day forecast",
    "recommend": "Where to focus",
    "whatif": "What-if scenario",
}

# When a clause asks for one specific metric, the heading should say that
# metric — "Recent performance" over a promo-uplift answer describes the intent
# label rather than what the section actually contains.
_METRIC_TITLES = {
    "promo_uplift": "Promotional uplift",
    "anomaly": "Anomalies",
    "drivers": "What is driving this",
    "trend": "Sales trend",
}


def _section_title(step: dict) -> str:
    # Only for performance steps: a forecast or what-if section that happens to
    # mention promotions is still a forecast, and titling both "Promotional
    # uplift" produced two identically-headed sections saying different things.
    if step["intent"] == "performance" and len(step.get("metrics") or []) == 1:
        specific = _METRIC_TITLES.get(step["metrics"][0])
        if specific:
            return specific
    return _SECTION_TITLES.get(step["intent"], step["intent"].title())


def _compose_fallback(state: AgentState, grounding: rv.Grounding) -> str:
    """Deterministic formatting — the grounded path.

    Used when no LLM is configured, when the LLM call fails, and whenever the
    LLM's own answer fails response validation. Every number it prints is
    registered with `grounding` as it is formatted, so it cannot state a figure
    the tools did not produce.
    """
    steps = state.get("steps") or []
    if not steps:
        return ("I couldn't find enough data to answer that. Try mentioning a specific "
                "store ID, e.g. 'How is Store 100 performing?'")

    sections: list[str] = []
    show_titles = len(steps) > 1
    for step in steps:
        builder = _SECTION_BUILDERS.get(step["intent"])
        body = builder(step, grounding) if builder else ""
        if not body:
            continue
        if show_titles:
            title = _section_title(step)
            scope = f" — Store{'s' if len(step['stores']) > 1 else ''} " + \
                    ", ".join(map(str, step["stores"])) if step["stores"] else " — fleet-wide"
            sections.append(f"### {title}{scope}\n\n{body}")
        else:
            sections.append(body)

    if not sections:
        return ("I couldn't find enough data to answer that. Try mentioning a specific "
                "store ID, e.g. 'How is Store 100 performing?'")

    return "\n\n".join(sections)


# ── LLM composition ──────────────────────────────────────────────────────────

_PROMPT_BY_INTENT = {
    "performance": prompts.SYSTEM_PROMPT_ANALYST,
    "forecast": prompts.SYSTEM_PROMPT_FORECAST,
    "recommend": prompts.SYSTEM_PROMPT_DECISION,
    "whatif": prompts.SYSTEM_PROMPT_WHATIF,
}


# Monthly history back to 2013 is 31 rows per store, and a writer needs the
# recent shape, not the archive.
_MAX_TREND_MONTHS = 3


def _llm_payload(step: dict) -> dict:
    """What the LLM needs to write this section, and nothing else.

    Strictly a subset/aggregation of `step["tool_results"]`, which is what the
    grounding is built from — so trimming the context can never let a figure
    through that the tools didn't produce. It only removes rows the model has
    no use for, which is both faster and less to go wrong with.
    """
    intent, tool_results = step["intent"], step["tool_results"]

    if intent == "performance":
        if step["stores"] and tool_results.get("store_metrics"):
            promo_history = tool_results.get("promo_history") or {}
            return {
                "stores": [
                    facts for facts in (
                        _store_performance_facts(sid, tool_results["store_metrics"],
                                                 promo_history.get(sid))
                        for sid in step["stores"]
                    ) if facts
                ],
                "recent_monthly_trend": {
                    sid: rows[-_MAX_TREND_MONTHS:]
                    for sid, rows in (tool_results.get("sales_trend") or {}).items()
                },
            }
        return {key: value for key, value in tool_results.items()
                if key in ("fleet_request", "sales_ranking", "promo_ranking", "eda_summary")}

    if intent == "forecast":
        forecasts = {}
        for sid, rows in (tool_results.get("forecast") or {}).items():
            values = [r["PredictedSales"] for r in rows]
            forecasts[sid] = {
                "daily": rows,
                "average_daily": round(sum(values) / len(values), 2) if values else 0,
                "window_total": round(sum(values), 2),
            }
        # baseline_comparison is charted on the Forecasting page, not narrated
        # here, so it is 21 rows of context the answer never refers to.
        return {"forecast": forecasts, "top_drivers": tool_results.get("shap") or {},
                "no_forecast": tool_results.get("no_forecast") or {}}

    if intent == "whatif":
        return {
            "whatif": {
                sid: {k: v for k, v in sim.items() if not k.endswith("_daily")}
                for sid, sim in (tool_results.get("whatif") or {}).items()
            },
            "no_forecast": tool_results.get("no_forecast") or {},
            "promo_history": tool_results.get("promo_history") or {},
        }

    if intent == "recommend":
        report = tool_results.get("decision_report") or {}
        return {
            "summary": report.get("summary"),
            "methodology": report.get("methodology"),
            "screened_stores": report.get("screened_stores"),
            "fleet_size": report.get("fleet_size"),
            "stores_without_forecast": report.get("stores_without_forecast"),
            "ranked_stores": [
                {k: v for k, v in entry.items() if k != "data_sources"}
                for entry in report.get("ranked_stores", [])
            ] or None,
            **{k: v for k, v in report.items()
               if k in ("observation", "prediction", "evidence", "recommendation",
                        "risk_level", "risk_score", "urgency")},
        }

    return tool_results


def _llm_context(state: AgentState) -> dict:
    """Everything the LLM is allowed to know: the tool output, and the limits
    the validation layer established. It gets no tools and no database."""
    result = state.get("validation")
    return {
        "question_parts": [
            {"asked": step["clause"], "intent": step["intent"], "stores": step["stores"]}
            for step in state.get("steps") or []
        ],
        "tool_results": [
            {"intent": step["intent"], "stores": step["stores"], "data": _llm_payload(step)}
            for step in state.get("steps") or []
        ],
        "must_tell_the_user": [f.message for f in (result.findings if result else [])],
        "dataset_coverage": result.coverage.as_dict() if result and result.coverage else {},
    }


# A single-section answer is asked to stay near 120 words; the rest is headroom.
# A stacked question legitimately needs more, so the budget grows per section
# rather than being one flat number that either truncates or invites rambling.
_TOKENS_PER_EXTRA_SECTION = 250
_MAX_COMPOSITION_TOKENS = 1400


def _token_budget(step_count: int) -> int:
    return min(LLM_MAX_TOKENS + max(0, step_count - 1) * _TOKENS_PER_EXTRA_SECTION,
               _MAX_COMPOSITION_TOKENS)


def _looks_truncated(text: str) -> bool:
    """A reply that stopped mid-sentence because it hit the token ceiling.

    Cheaper to detect than to prevent, and the grounded template is always
    available — so a truncated answer is discarded rather than shown.
    """
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] not in ".!?:)]\"'*`%…"


def _compose_with_llm(state: AgentState, context: dict) -> str:
    _emit("Writing the answer")
    llm = _get_llm()
    budget = _token_budget(len(state.get("steps") or []))
    try:
        llm = llm.bind(max_tokens=budget)
    except Exception:  # pragma: no cover - provider without bind support
        logger.debug("could not bind a per-question token budget", exc_info=True)
    system_prompt = _PROMPT_BY_INTENT.get(state.get("intent"), prompts.SYSTEM_PROMPT_DECISION)
    if len(state.get("steps") or []) > 1:
        system_prompt = f"{prompts.SYSTEM_PROMPT_MULTI_INTENT}\n\n{system_prompt}"
    context_json = json.dumps(context, default=str, indent=2)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"User question: {state['query']}\n\n"
            f"Tool results (the ONLY data you may reference):\n{context_json}"
        )),
    ]
    return llm.invoke(messages).content


def _build_grounding(state: AgentState) -> rv.Grounding:
    grounding = rv.Grounding()
    for step in state.get("steps") or []:
        grounding.ingest(step["tool_results"])
        for sid in step["stores"]:
            grounding.add_store(sid)
    result = state.get("validation")
    if result:
        for verdict in result.claim_verdicts:
            if verdict.actual_value is not None:
                grounding.add_number(verdict.actual_value)
        # Figures quoted inside a finding's own message are grounded — they were
        # computed from the data by the validation layer.
        for finding in result.findings:
            for token in re.findall(r"-?[\d,]+(?:\.\d+)?", finding.message):
                grounding.add_number(token.replace(",", ""))
        for store_id in result.known_stores + result.unknown_stores:
            grounding.add_store(store_id)
    return grounding


def respond_node(state: AgentState) -> dict:
    """Compose the answer, then check it before letting it out."""
    result = state.get("validation")
    grounding = _build_grounding(state)
    intent_types = [step["intent"] for step in state.get("steps") or []]
    # Blocking findings are surfaced here too, not only by refuse_node: when a
    # stacked question has one unanswerable part, the answer has to say which
    # part it refused as well as answering the rest.
    notices = [f.message for f in (result.findings if result else [])]
    if state.get("guardrail_notice"):
        # Part of the message was refused. The refusal leads, then the part
        # that could legitimately be answered follows.
        notices.insert(0, state["guardrail_notice"])
    contradicted = [v.claim.raw for v in (result.claim_verdicts if result else [])
                    if v.status == "contradicted"]
    sources = state.get("data_sources", [])
    coverage = result.coverage.as_dict() if result and result.coverage else None
    error = state.get("error")

    def _check(text: str) -> rv.ValidationReport:
        return rv.validate_response(
            text, grounding,
            intent_types=intent_types,
            required_notices=notices,
            expected_sources=sources,
            contradicted_claims=contradicted,
            coverage=coverage,
        )

    response = None
    try:
        candidate = _compose_with_llm(state, _llm_context(state))
        if _looks_truncated(candidate):
            raise ValueError("model reply was cut off at the token limit")
        report = _check(_assemble(candidate, notices, sources))
        if report.passed:
            response = candidate
        else:
            # The model's wording failed a mechanical check against the real
            # data, so it doesn't get to be the answer.
            logger.warning("LLM response failed validation, using grounded template: %s",
                           report.summary())
            error = f"{error + ' | ' if error else ''}response_validation: {report.summary()}"
    except Exception as exc:
        logger.warning("LLM composition failed, using deterministic template: %s", exc)
        error = f"{error + ' | ' if error else ''}compose_llm: {exc}"

    if response is None:
        response = _compose_fallback(state, grounding)

    final = _assemble(response, notices, sources)

    report = _check(final)
    if not report.passed:
        # The grounded path failing means a real defect, not a wording problem.
        logger.error("final response failed validation: %s", report.summary())
        error = f"{error + ' | ' if error else ''}final_validation: {report.summary()}"

    # Last line of defence: the LLM's wording isn't fully predictable, so
    # anything credential-shaped is stripped.
    return {"response": guardrails.redact_output(final), "error": error, "grounding": grounding}


# Matches a citation block to the end of the text, with or without the emoji.
_SOURCES_BLOCK_RE = re.compile(r"\n*(?:📚\s*)?\**\s*(?:Data\s+)?Sources?\s*:?\**\s*\n.*\Z",
                               re.IGNORECASE | re.DOTALL)


def _assemble(body: str, notices: list[str], sources: list[str]) -> str:
    """Notices first, then the answer, then citations.

    Notices lead because they change how the rest should be read — a correction
    to the user's premise, or a boundary on what was analysed, is useless
    underneath the answer it qualifies.
    """
    parts = []
    if notices:
        parts.append("\n".join(f"> ⚠️ {n}" for n in notices))
    parts.append(body)
    text = "\n\n".join(p for p in parts if p)

    # Any citation block the model wrote itself is discarded and replaced with
    # the functions that actually ran. Asked to "cite the sources you were
    # given", a model will cite whatever it sees — one run cited the JSON key
    # "tool_results" as though it were a data source, and because the text then
    # contained the word "Sources" the real list was suppressed. Citations are
    # a record of what executed, so they are written from the call log, never
    # by the model.
    text = _SOURCES_BLOCK_RE.sub("", text).rstrip()

    if sources:
        # One source per line, not comma-separated — pages/4_AI_Assistant.py
        # (Dikshit) splits this block on newlines to build separate citation
        # cards; a single comma-joined line renders as one garbled citation.
        sources_block = "\n".join(f"- {s}" for s in sources)
        text += f"\n\n📚 Sources:\n{sources_block}"
    return text


# ── Graph assembly ───────────────────────────────────────────────────────────

def _route_after_guardrail(state: AgentState) -> str:
    return "blocked" if state.get("blocked_reason") else "understand"


def _route_after_validation(state: AgentState) -> str:
    result = state.get("validation")
    return "execute" if result and result.is_answerable else "refuse"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("understand", understand_node)
    graph.add_node("validate", validate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("respond", respond_node)

    # Screening comes first, and a blocked message goes straight to END —
    # it never reaches the understanding layer, the database, or the model.
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail", _route_after_guardrail,
        {"understand": "understand", "blocked": END},
    )
    graph.add_edge("understand", "validate")
    graph.add_conditional_edges(
        "validate", _route_after_validation,
        {"execute": "execute", "refuse": "refuse"},
    )
    graph.add_edge("execute", "respond")
    # Straight to END: the refusal is already the final answer, and sending it
    # through respond would hand an unanswerable question to the LLM.
    graph.add_edge("refuse", END)
    graph.add_edge("respond", END)

    return graph.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def check_llm_connection() -> dict:
    """Diagnostic: is the OpenRouter key/model actually reachable right now?

    The agent degrades to a deterministic template when the LLM fails, which
    looks like a working answer — so run this to tell "LLM is working" apart
    from "LLM silently fell back". Returns {ok, model, detail}.
    """
    if not LLM_API_KEY:
        return {"ok": False, "model": LLM_MODEL,
                "detail": "OPENROUTER_API_KEY is not set (check your .env file)"}

    try:
        reply = _get_llm().invoke([HumanMessage(content="Reply with the single word: ok")])
        return {"ok": True, "model": LLM_MODEL, "detail": (reply.content or "").strip()[:80]}
    except Exception as exc:
        return {"ok": False, "model": LLM_MODEL, "detail": f"{type(exc).__name__}: {exc}"}


USER_FACING_ERROR = (
    "I couldn't process that request. Please try asking about store sales, "
    "forecasts, promotions, or which stores need attention."
)


def run_agent(user_query: str, session_id: str = "default") -> str:
    """Synchronous agent call."""
    try:
        result = _get_graph().invoke({"query": user_query, "session_id": session_id})
        if result.get("error"):
            # Kept server-side only; the user sees the answer, not the plumbing.
            logger.warning("agent run completed with degraded path: %s", result["error"])
        return result.get("response") or USER_FACING_ERROR
    except Exception:
        # exc_info goes to the log, never into the returned string: a raw
        # exception can carry SQL, file paths and schema details.
        logger.exception("agent run failed for session %s", session_id)
        return USER_FACING_ERROR


def run_agent_stream(user_query: str, session_id: str = "default") -> Generator[str, None, None]:
    """Streaming version for Streamlit chat UI."""
    try:
        response = run_agent(user_query, session_id)
    except Exception:
        logger.exception("agent stream failed for session %s", session_id)
        yield USER_FACING_ERROR
        return

    words = response.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")

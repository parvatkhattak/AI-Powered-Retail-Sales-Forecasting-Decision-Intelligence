"""
src/agent_graph.py
Owner: Saumya — Agentic AI Engineer

Responsibilities:
- Define LangGraph state and nodes
- Route user queries
- Execute tool calls
- Format final responses

Graph design (see docs/architecture.md, section 6.2):

    START -> Router -> {DataAnalyst | Forecast | Decision} -> Respond -> END

The Router classifies intent and extracts store IDs; exactly one
downstream node runs real tool calls against database.py / model_engine.py
/ decision_engine.py; Respond turns the tool results into the final
markdown answer (LLM-composed when an API key is configured, otherwise a
deterministic template so the agent never crashes without one).
"""

import json
import re
import sys
from pathlib import Path
from typing import Generator, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from src import database, decision_engine, model_engine, prompts

Intent = Literal["performance", "forecast", "recommend", "whatif"]

# Only numbers that follow the word "store(s)" count as store IDs — avoids
# false positives like "top 10 stores" or "next 7 days" (rule: never hardcode
# a store ID, always extract dynamically from the query).
_STORE_MENTION_RE = re.compile(
    r"stores?\s*(?:id[s]?)?\s*[:#]?\s*((?:(?:\s*(?:,|and|&|/)\s*)*\d+\s*)+)", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"\d+")
_MIN_STORE_ID, _MAX_STORE_ID = 1, 1115


class AgentState(TypedDict, total=False):
    query: str                    # Original user question
    session_id: str                # For conversation memory
    intent: Intent                # "performance" | "forecast" | "recommend" | "whatif"
    store_ids: list[int]          # Extracted store IDs from query
    tool_results: dict            # Raw data returned by tool calls
    decision_report: dict         # Structured output from decision_engine
    response: str                 # Final formatted response for the user
    data_sources: list[str]       # Citation list for UI citation cards
    error: str | None             # Error message if something fails


class RouterOutput(BaseModel):
    intent: Intent = Field(description="The single best-matching intent for the user's question.")


def _extract_store_ids(query: str) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for mention in _STORE_MENTION_RE.finditer(query):
        for num in _NUMBER_RE.findall(mention.group(1)):
            sid = int(num)
            if _MIN_STORE_ID <= sid <= _MAX_STORE_ID and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def _classify_intent_fallback(query: str, store_ids: list[int]) -> Intent:
    """Keyword-based classifier used when no LLM is available (per docs/architecture.md
    6.3: "keyword matching + LLM classification")."""
    q = query.lower()
    if "promo" in q and any(k in q for k in ("what if", "what-if", "simulate", "toggle", " if we", " if they")):
        return "whatif"
    # Checked before "forecast" keywords: a multi-store or "which should I focus
    # on" question usually also mentions the forecast horizon, but it's asking
    # for a ranked recommendation, not a plain forecast.
    if any(k in q for k in ("focus", "recommend", "priorit", "should i", "compare")) or len(store_ids) > 1:
        return "recommend"
    if any(k in q for k in ("forecast", "predict", "expect", "next week", "next 7 days")):
        return "forecast"
    return "performance"


_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        if not LLM_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set — add it to your .env file.")
        _llm = ChatOpenRouter(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=LLM_API_KEY)
    return _llm


# ── Nodes ────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    query = state["query"]
    store_ids = _extract_store_ids(query)

    try:
        structured_llm = _get_llm().with_structured_output(RouterOutput)
        decision = structured_llm.invoke([
            SystemMessage(content=prompts.SYSTEM_PROMPT_ROUTER),
            HumanMessage(content=query),
        ])
        intent = decision.intent
    except Exception:
        intent = _classify_intent_fallback(query, store_ids)

    return {"store_ids": store_ids, "intent": intent, "error": None}


def data_analyst_node(state: AgentState) -> dict:
    """Tools: get_store_metrics, get_promo_history, get_sales_trend (database.py)."""
    store_ids = state.get("store_ids", [])
    tool_results: dict = {}
    sources: list[str] = []

    if store_ids:
        metrics_df = database.get_store_metrics(store_ids, days=30)
        tool_results["store_metrics"] = metrics_df.to_dict(orient="records")
        tool_results["promo_history"] = {sid: database.get_promo_history(sid) for sid in store_ids}
        tool_results["sales_trend"] = {
            sid: database.get_sales_trend(sid).to_dict(orient="records") for sid in store_ids
        }
        sources = ["database.get_store_metrics", "database.get_promo_history", "database.get_sales_trend"]
    else:
        tool_results["eda_summary"] = database.get_eda_summary()
        sources = ["database.get_eda_summary"]

    return {"tool_results": tool_results, "data_sources": sources}


def forecast_node(state: AgentState) -> dict:
    """Tools: get_7day_forecast, get_shap_explanations, get_baseline_comparison (model_engine.py)."""
    store_ids = state.get("store_ids", [])
    tool_results: dict = {"forecast": {}, "shap": {}, "baseline_comparison": {}}

    for sid in store_ids:
        tool_results["forecast"][sid] = model_engine.get_7day_forecast(sid).to_dict(orient="records")
        tool_results["shap"][sid] = model_engine.get_shap_explanations(sid)
        tool_results["baseline_comparison"][sid] = model_engine.get_baseline_comparison(sid).to_dict(orient="records")

    sources = (
        ["model_engine.get_7day_forecast", "model_engine.get_shap_explanations", "model_engine.get_baseline_comparison"]
        if store_ids
        else []
    )
    return {"tool_results": tool_results, "data_sources": sources}


def decision_node(state: AgentState) -> dict:
    """Calls decision_engine.py, which internally fetches from both database.py and
    model_engine.py (per docs/architecture.md 6.3, Decision Node)."""
    store_ids = state.get("store_ids", [])

    if len(store_ids) > 1:
        report = decision_engine.compare_stores_report(store_ids)
        sources = sorted({s for r in report["ranked_stores"] for s in r["data_sources"]})
    elif len(store_ids) == 1:
        report = decision_engine.generate_decision_report(store_ids[0])
        sources = report["data_sources"]
    else:
        report = {"summary": "No specific store was identified in the question.", "ranked_stores": []}
        sources = []

    return {"decision_report": report, "data_sources": sources}


_PROMPT_BY_INTENT = {
    "performance": prompts.SYSTEM_PROMPT_ANALYST,
    "forecast": prompts.SYSTEM_PROMPT_FORECAST,
    "recommend": prompts.SYSTEM_PROMPT_DECISION,
    "whatif": prompts.SYSTEM_PROMPT_WHATIF,
}


def _compose_with_llm(state: AgentState, context: dict) -> str:
    llm = _get_llm()
    system_prompt = _PROMPT_BY_INTENT.get(state.get("intent"), prompts.SYSTEM_PROMPT_DECISION)
    context_json = json.dumps(context, default=str, indent=2)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User question: {state['query']}\n\nTool results (the ONLY data you may reference):\n{context_json}"),
    ]
    return llm.invoke(messages).content


def _compose_fallback(state: AgentState) -> str:
    """Deterministic formatting used when the LLM is unavailable — keeps the agent
    testable and crash-free without an API key (docs/architecture.md 13, "Agent
    Error Handling")."""
    store_ids = state.get("store_ids", [])
    report = state.get("decision_report")
    tool_results = state.get("tool_results", {})

    if report and report.get("ranked_stores"):
        lines = [f"**Priority ranking for stores {', '.join(map(str, store_ids))}:**\n"]
        for i, r in enumerate(report["ranked_stores"], start=1):
            lines.append(
                f"{i}. **Store {r['store_id']}** — risk: {r['risk_level']}\n"
                f"   - 🔍 Observation: {r['observation']}\n"
                f"   - 📈 Prediction: {r['prediction']}\n"
                f"   - 📊 Evidence: {r['evidence']}\n"
                f"   - ✅ Recommendation: {r['recommendation']}"
            )
        lines.append(f"\n**Summary:** {report['summary']}")
        return "\n".join(lines)

    if report and "observation" in report:
        return (
            f"🔍 **Observation:** {report['observation']}\n\n"
            f"📈 **Prediction:** {report['prediction']}\n\n"
            f"📊 **Evidence:** {report['evidence']}\n\n"
            f"✅ **Recommendation:** {report['recommendation']}"
        )

    if tool_results.get("eda_summary"):
        s = tool_results["eda_summary"]
        return (
            f"Across {s.get('total_stores', 'N/A')} stores, average daily sales are "
            f"€{s.get('avg_daily_sales', 0):,.0f}. Store {s.get('best_store_id')} performs best "
            f"(€{s.get('best_store_avg_sales', 0):,.0f}/day) and Store {s.get('worst_store_id')} "
            f"performs worst (€{s.get('worst_store_avg_sales', 0):,.0f}/day)."
        )

    if tool_results.get("forecast"):
        lines = []
        for sid, rows in tool_results["forecast"].items():
            if rows:
                avg = sum(r["PredictedSales"] for r in rows) / len(rows)
                lines.append(f"Store {sid}: 7-day forecast averages €{avg:,.0f}/day.")
        return "\n".join(lines) if lines else "No forecast data available for the requested store(s)."

    if tool_results.get("store_metrics"):
        return f"Retrieved {len(tool_results['store_metrics'])} days of sales history for store(s) {', '.join(map(str, store_ids))}."

    return "I couldn't find enough data to answer that. Try mentioning a specific store ID, e.g. 'How is Store 100 performing?'"


def respond_node(state: AgentState) -> dict:
    context = state.get("decision_report") or state.get("tool_results") or {}

    try:
        response = _compose_with_llm(state, context)
    except Exception:
        response = _compose_fallback(state)

    sources = state.get("data_sources", [])
    if sources and "sources" not in response.lower():
        response += "\n\n📚 **Sources:** " + ", ".join(sources)

    return {"response": response}


def _route_from_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "forecast":
        return "forecast"
    if intent in ("recommend", "whatif"):
        # Full What-If simulation node is an S-grade / Day-3 addition (see
        # docs/architecture.md 6.3); until then, whatif questions fall back to
        # the Decision node, which still grounds its answer in real forecasts.
        return "decision"
    return "data_analyst"


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("decision", decision_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_from_intent,
        {"data_analyst": "data_analyst", "forecast": "forecast", "decision": "decision"},
    )
    graph.add_edge("data_analyst", "respond")
    graph.add_edge("forecast", "respond")
    graph.add_edge("decision", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(user_query: str, session_id: str = "default") -> str:
    """Synchronous agent call."""
    try:
        result = _get_graph().invoke({"query": user_query, "session_id": session_id})
        return result.get("response") or "Sorry, I couldn't generate a response for that question."
    except Exception as exc:
        return f"⚠️ Something went wrong while processing your question: {exc}"


def run_agent_stream(user_query: str, session_id: str = "default") -> Generator[str, None, None]:
    """Streaming version for Streamlit chat UI."""
    try:
        response = run_agent(user_query, session_id)
    except Exception as exc:
        yield f"⚠️ Something went wrong while processing your question: {exc}"
        return

    words = response.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")

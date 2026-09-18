"""
pages/6_📡_Observability.py
Owner: Parvat — Team Lead / Integration

Live monitoring dashboard showing:
  - System health (DB, models, API key, runtime)
  - FinOps: token usage, estimated cost, LLM call log
  - Performance: latency trends for forecast + agent calls
  - Error log
  - Structured log tail
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Observability | Retail AI",
    page_icon="📡",
    layout="wide",
)

# ── Safe import of observability module ───────────────────────────────────────
try:
    from src.observability import (
        get_log_tail,
        get_session_summary,
        run_health_check,
    )
    _OBS_AVAILABLE = True
except Exception as exc:
    _OBS_AVAILABLE = False
    _OBS_ERROR = str(exc)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📡 Observability & FinOps Dashboard")
st.caption("Live monitoring · Token tracking · Cost estimation · Latency · Health checks")

if not _OBS_AVAILABLE:
    st.error(f"Observability module failed to load: {_OBS_ERROR}")
    st.stop()

# ── Auto-refresh toggle ───────────────────────────────────────────────────────
col_title, col_refresh = st.columns([6, 1])
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🟢 System Health")

_STATUS_ICON = {"ok": "✅", "warn": "⚠️", "error": "❌"}
_STATUS_COLOR = {"ok": "green", "warn": "orange", "error": "red"}

with st.spinner("Running health checks…"):
    health = run_health_check()

health_cols = st.columns(len(health))
for col, (subsystem, info) in zip(health_cols, health.items()):
    icon   = _STATUS_ICON.get(info["status"], "❓")
    color  = _STATUS_COLOR.get(info["status"], "grey")
    label  = subsystem.replace("_", " ").title()
    with col:
        st.markdown(
            f"""
            <div style="border:1px solid {color};border-radius:10px;padding:14px;text-align:center;background:#111;">
                <div style="font-size:28px">{icon}</div>
                <div style="font-weight:700;margin:6px 0 4px;color:#eee">{label}</div>
                <div style="font-size:11px;color:#aaa">{info['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SESSION SUMMARY KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Session Summary")

summary = get_session_summary()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🤖 LLM Calls",     summary.get("total_llm_calls",  0))
k2.metric("🔤 Tokens Used",   f"{summary.get('total_tokens',  0):,}")
k3.metric("💰 Est. Cost",     f"${summary.get('total_cost_usd', 0):.6f}")
k4.metric("⚡ Avg Latency",   f"{summary.get('avg_latency_ms', 0):.0f} ms")
k5.metric("🚨 Errors",        summary.get("error_count", 0))

st.caption(f"Session ID: `{summary.get('session_id', 'n/a')}`")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FINOPS: LLM CALL LOG
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💰 FinOps — LLM Token & Cost Log")

llm_calls = summary.get("llm_calls", [])
if not llm_calls:
    st.info("No LLM calls recorded in this session yet. Use the 🤖 AI Assistant to generate some.")
else:
    import pandas as pd

    df_llm = pd.DataFrame(llm_calls)[[
        "ts", "model", "prompt_tokens", "completion_tokens",
        "total_tokens", "cost_usd", "latency_ms", "status", "query_preview",
    ]].rename(columns={
        "ts": "Timestamp",
        "model": "Model",
        "prompt_tokens": "Prompt Tok",
        "completion_tokens": "Completion Tok",
        "total_tokens": "Total Tok",
        "cost_usd": "Cost (USD)",
        "latency_ms": "Latency (ms)",
        "status": "Status",
        "query_preview": "Query Preview",
    })
    # Format timestamp to readable
    df_llm["Timestamp"] = pd.to_datetime(df_llm["Timestamp"]).dt.strftime("%H:%M:%S")

    st.dataframe(df_llm, use_container_width=True, hide_index=True)

    # Token usage chart
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        idx = list(range(1, len(df_llm) + 1))
        fig.add_bar(x=idx, y=df_llm["Prompt Tok"],     name="Prompt tokens",     marker_color="#4e8ef7")
        fig.add_bar(x=idx, y=df_llm["Completion Tok"], name="Completion tokens", marker_color="#f77c4e")
        fig.update_layout(
            barmode="stack",
            title="Token Usage per LLM Call",
            xaxis_title="Call #",
            yaxis_title="Tokens",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="#eee",
            legend=dict(orientation="h", y=-0.2),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LATENCY TRENDS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("⚡ Performance — Latency Trends")

latency_events = summary.get("latency_events", [])
if not latency_events:
    st.info("No latency events recorded yet. Run a forecast or ask the AI Assistant.")
else:
    import pandas as pd
    import plotly.graph_objects as go

    df_lat = pd.DataFrame(latency_events)
    df_lat["ts"] = pd.to_datetime(df_lat["ts"])

    # One line per event type
    fig = go.Figure()
    for event_name in df_lat["event"].unique():
        sub = df_lat[df_lat["event"] == event_name].sort_values("ts")
        fig.add_scatter(
            x=sub["ts"], y=sub["latency_ms"],
            mode="lines+markers", name=event_name,
        )
    fig.update_layout(
        title="Latency Over Time (ms)",
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#eee",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    summary_lat = (
        df_lat.groupby("event")["latency_ms"]
        .agg(calls="count", mean="mean", p95=lambda x: x.quantile(0.95), max="max")
        .round(1)
        .reset_index()
        .rename(columns={"event": "Event", "calls": "Calls", "mean": "Mean (ms)", "p95": "P95 (ms)", "max": "Max (ms)"})
    )
    st.dataframe(summary_lat, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ERROR LOG
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🚨 Error Log")

errors = summary.get("errors", [])
if not errors:
    st.success("No errors recorded in this session. ✅")
else:
    import pandas as pd
    df_err = pd.DataFrame(errors)
    df_err["ts"] = pd.to_datetime(df_err["ts"]).dt.strftime("%H:%M:%S")
    st.dataframe(
        df_err.rename(columns={"ts": "Time", "event": "Event", "message": "Message"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — LIVE LOG TAIL
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📜 Structured Log Tail")

n_lines = st.slider("Lines to show", min_value=10, max_value=100, value=30, step=10)
log_entries = get_log_tail(n=n_lines)

if not log_entries:
    st.info("No log entries yet. Log file will be created in `logs/app.log` once the app runs some operations.")
else:
    import pandas as pd
    df_log = pd.DataFrame(log_entries)

    # Keep only interesting columns if they exist
    keep = [c for c in ["ts", "level", "logger", "msg", "event", "latency_ms", "total_tokens", "status"] if c in df_log.columns]
    st.dataframe(df_log[keep], use_container_width=True, hide_index=True)

    # Raw JSON expander
    with st.expander("🔍 Raw JSON log"):
        import json
        st.code(
            "\n".join(json.dumps(e, default=str) for e in log_entries[:20]),
            language="json",
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.caption(
    "📡 **Observability** · Logs → `logs/app.log` · FinOps → `logs/finops.jsonl` · "
    "Latency → `logs/metrics.jsonl`"
)

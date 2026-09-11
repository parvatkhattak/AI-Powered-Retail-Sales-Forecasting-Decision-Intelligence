"""
pages/1_🏠_Dashboard.py
Owner: Dikshit — Frontend Developer

Executive Dashboard:
- KPI summary cards (total stores, avg daily sales, best/worst store)
- Sales trend chart (all stores, monthly)
- Store ranking table (top 10 by avg sales)
- Anomaly alert section
- CSV export of store metrics
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui_theme import apply_theme
from src import database

st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide")
apply_theme()

st.markdown("""
<style>
    p, li { font-size: 1.1rem; }
    [data-testid="stMetricValue"] { font-size: 2.4rem !important; }
    [data-testid="stMetricLabel"] { font-size: 1.1rem !important; }
    [data-testid="stMetricDelta"] { font-size: 1.0rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("# 🏠 Executive Dashboard")
st.markdown("**Live overview of all 1,115 Rossmann stores — sales performance, trends, and anomalies.**")
st.divider()

# ── Loading spinner ───────────────────────────────────────────────────────────
with st.spinner("Loading store metrics…"):
    try:
        eda = database.get_eda_summary()
        stores_df = database.get_all_stores()
        promo_ranking = database.get_promo_uplift_ranking(top_n=10)
        anomalies = database.get_anomaly_flags(1, lookback_days=90)
        data_loaded = True
    except Exception as e:
        st.error(f"⚠️ Could not load data: {e}\n\nRun `python3 src/data_pipeline.py` to initialise the database.")
        data_loaded = False

if not data_loaded:
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown("### 📊 Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)

total_stores = eda.get("total_stores", "N/A")
avg_daily = eda.get("avg_daily_sales", 0)
best_id = eda.get("best_store_id", "N/A")
best_sales = eda.get("best_store_avg_sales", 0)
worst_id = eda.get("worst_store_id", "N/A")
worst_sales = eda.get("worst_store_avg_sales", 0)

k1.metric("🏪 Total Stores", f"{total_stores:,}" if isinstance(total_stores, int) else total_stores)
k2.metric("💶 Avg Daily Sales", f"€{avg_daily:,.0f}" if avg_daily else "N/A")
k3.metric("🥇 Best Store", f"Store {best_id}", delta=f"€{best_sales:,.0f}/day" if best_sales else None)
k4.metric("🔴 Weakest Store", f"Store {worst_id}", delta=f"€{worst_sales:,.0f}/day" if worst_sales else None, delta_color="inverse")

st.divider()

# ── Sales Trend ───────────────────────────────────────────────────────────────
st.markdown("### 📈 Chain-Wide Sales Trend (Store 1 — Last 2 Years)")
with st.spinner("Loading sales trend…"):
    try:
        trend_df = database.get_sales_trend(1, period="monthly")
    except Exception:
        trend_df = pd.DataFrame()

if not trend_df.empty:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_df["period"], y=trend_df["avg_sales"],
        mode="lines+markers", name="Avg Daily Sales",
        line=dict(color="#10b981", width=3),
        marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
    ))
    fig_trend.update_layout(
        xaxis_title="Period", yaxis_title="Avg Daily Sales (€)",
        template="plotly_dark", height=380, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14, color="#e5e7eb"),
    )
    st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")
else:
    st.info("Sales trend data not available — run `python3 src/data_pipeline.py` first.")

st.divider()

# ── Store Rankings & Promo Uplift ─────────────────────────────────────────────
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### 🏆 Top 10 Stores by Promo Sales Uplift")
    if not promo_ranking.empty:
        fig_promo = px.bar(
            promo_ranking,
            x="uplift_pct", y=promo_ranking["Store"].astype(str),
            orientation="h",
            color="uplift_pct",
            color_continuous_scale=["#1f2937", "#10b981"],
            labels={"uplift_pct": "Promo Uplift (%)", "y": "Store ID"},
        )
        fig_promo.update_layout(
            template="plotly_dark", height=380, margin=dict(t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, showlegend=False,
            font=dict(size=13, color="#e5e7eb"),
        )
        st.plotly_chart(fig_promo, use_container_width=True, key="promo_chart")
    else:
        st.info("Promo uplift data unavailable.")

with right_col:
    st.markdown("### 🏪 Store Type Breakdown")
    if not stores_df.empty and "StoreType" in stores_df.columns:
        type_counts = stores_df["StoreType"].value_counts().reset_index()
        type_counts.columns = ["StoreType", "Count"]
        fig_pie = px.pie(
            type_counts, values="Count", names="StoreType",
            color_discrete_sequence=["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b"],
        )
        fig_pie.update_layout(
            template="plotly_dark", height=380, margin=dict(t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=14, color="#e5e7eb"),
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="type_pie")
    else:
        st.info("Store type data unavailable.")

st.divider()

# ── Store Metrics Table with CSV export ───────────────────────────────────────
st.markdown("### 📋 Full Store Metrics Table")

if not stores_df.empty:
    st.dataframe(
        stores_df.style.highlight_max(subset=["CompetitionDistance"], color="#1e3a2f"),
        use_container_width=True, height=350,
    )

    # CSV Export
    csv_bytes = stores_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Store Metrics as CSV",
        data=csv_bytes,
        file_name="all_stores_metrics.csv",
        mime="text/csv",
        use_container_width=False,
    )
else:
    st.info("Store data unavailable — run `python3 src/data_pipeline.py` first.")

st.divider()

# ── Anomaly Alerts ────────────────────────────────────────────────────────────
st.markdown("### 🚨 Recent Anomaly Alerts (Store 1)")
if not anomalies.empty:
    st.dataframe(
        anomalies.style.applymap(
            lambda v: "background-color: #7f1d1d;" if isinstance(v, float) and abs(v) > 3 else "",
            subset=["sales_zscore"],
        ),
        use_container_width=True,
    )
else:
    st.success("✅ No anomalies detected in the last 90 days for Store 1.")

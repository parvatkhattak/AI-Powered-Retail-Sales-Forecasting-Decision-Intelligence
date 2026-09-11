"""
pages/2_📈_Forecasting.py
Owner: Ashutosh — ML Engineer / Dikshit — UI Developer

7-Day Sales Forecasting page:
- Recursive LightGBM forecast with confidence intervals
- Store selector and feature highlights
- SHAP feature importance driver bar chart & waterfall breakdown
- What-If promo impact simulation
- Multi-model agreement chart (LightGBM vs XGBoost vs Naive Baseline)
- Automated natural language forecast narrative & CSV export
"""

import sys
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DB_PATH, USE_MOCKS
from src.model_engine import (
    get_7day_forecast, get_shap_explanations, get_missing_store_info,
    get_shap_waterfall_data, get_baseline_comparison, get_whatif_forecast,
    get_forecast_calendar,
)

st.set_page_config(
    page_title="Sales Forecasting — Retail AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (matches Dashboard dark theme) ─────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0c14 0%, #0e1117 50%, #0a0f1e 100%); }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2133 100%);
        border: 1px solid #2E3250; border-radius: 16px; padding: 20px 24px;
        box-shadow: 0 4px 24px rgba(108,99,255,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px); box-shadow: 0 8px 32px rgba(108,99,255,0.18);
    }
    div[data-testid="metric-container"] > label {
        font-size: 0.78rem !important; font-weight: 600 !important;
        letter-spacing: 0.06em; text-transform: uppercase; color: #8B8FA8 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem !important; font-weight: 700 !important; color: #E0E0FF !important;
    }
    hr { border-color: #2E3250 !important; margin: 2rem 0 !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #141727 100%);
        border-right: 1px solid #2E3250;
    }
    div[data-testid="stPlotlyChart"] > div {
        border-radius: 12px; border: 1px solid #2E3250; overflow: hidden;
    }
    .page-title {
        background: linear-gradient(90deg, #6C63FF, #43D9A4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 700; margin-bottom: 0.1rem;
    }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
             font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }
    .badge-live { background: rgba(67,217,164,0.15); color: #43D9A4; border: 1px solid #43D9A4; }
    .badge-mock { background: rgba(255,165,82,0.15); color: #FFA552; border: 1px solid #FFA552; }
    .hero-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2133 100%);
        border: 1px solid #2E3250; border-radius: 16px; padding: 24px;
        margin-bottom: 1.5rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #141727 0%, #1a1d2e 100%);
        border: 1px solid #2E3250; border-radius: 14px; padding: 18px 20px;
        margin-bottom: 12px; display: flex; align-items: center; gap: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
mode_badge = '<span class="badge badge-mock">MOCK MODE</span>' if USE_MOCKS else '<span class="badge badge-live">LIVE DB</span>'
st.markdown(
    f'<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem;">'
    f'<span class="page-title">📈 7-Day Sales Forecasting</span>{mode_badge}</div>',
    unsafe_allow_html=True,
)
st.caption("Generate multi-step LightGBM forecasts with confidence intervals, scenario simulation, and model explanations.")
st.divider()

# ── Feature Dictionary ────────────────────────────────────────────────────────
FEATURE_PHRASES = {
    "Promo": "today's promotion", "Sales_roll_mean_7": "this store's average sales over the last 7 days",
    "Sales_roll_mean_14": "this store's average sales over the last 14 days",
    "Sales_roll_mean_28": "this store's average sales over the last 28 days",
    "Sales_roll_std_7": "how much sales have swung around lately", "Sales_roll_std_14": "how much sales have swung around lately",
    "Sales_roll_std_28": "how much sales have swung around lately",
    "Sales_lag_7": "sales on this same weekday a week ago", "Sales_lag_14": "sales two weeks ago on this weekday",
    "Sales_lag_28": "sales four weeks ago on this weekday",
    "DayOfWeek": "which day of the week it is", "Day": "the day of the month",
    "Month": "the time of year", "WeekOfYear": "the particular week of the year",
    "Quarter": "the quarter of the year", "IsWeekend": "the weekend",
    "StateHoliday": "a public holiday", "SchoolHoliday": "school holidays",
    "PromoStreak": "how long this promotion has been running",
    "DaysSinceLastPromo": "how recently the last promotion ended",
    "DaysUntilNextPromo": "how soon the next promotion starts",
    "IsPromo2Active": "the recurring quarterly promotion",
    "CompetitionDistance": "how close the nearest competitor is",
    "CompetitionOpenMonths": "how long the nearby competitor has existed",
    "StoreType": "this store's format", "Assortment": "this store's product range",
    "store_cluster": "the group of similar stores this one belongs to",
    "Promo2": "the recurring quarterly promotion program",
}


def top_driver_phrase(shap_dict: dict) -> str:
    if not shap_dict:
        return "its usual seasonal pattern"
    name, info = next(iter(shap_dict.items()))
    return FEATURE_PHRASES.get(name, name)


def store_baseline(store_id: int) -> float:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT AVG(Sales) FROM sales WHERE Store = ? AND Open = 1", (store_id,)).fetchone()
    conn.close()
    return float(row[0]) if row and row[0] else 0.0


def _day_reason(date, pct: float, calendar, next_trading_date, driver_phrase) -> str:
    promo_today = False
    if calendar is not None and not calendar.empty:
        cal_row = calendar[calendar["Date"] == date]
        if not cal_row.empty:
            promo_today = bool(cal_row.iloc[0]["Promo"])
    if pct >= 0 and promo_today:
        return "a promotion running that day"
    if pct < 0 and not promo_today:
        return "no promotion running that day"
    if next_trading_date is not None and date == next_trading_date:
        return driver_phrase
    return "this store's typical rhythm for that day of the week"


def _magnitude_word(pct: float) -> str:
    a = abs(pct)
    if a >= 20:
        return "significantly"
    if a >= 10:
        return "notably"
    if a >= 4:
        return "modestly"
    return "only slightly"


def build_narrative(store_id: int, forecast, calendar=None) -> str:
    if calendar is None:
        calendar = pd.DataFrame(columns=["Date", "Promo"])
    open_days = forecast[forecast["PredictedSales"] > 0]
    baseline = store_baseline(store_id)
    week_total = int(open_days["PredictedSales"].sum())

    lines = [
        f"Over the next 7 days, Store {store_id} is expected to bring in around "
        f"<b>€{week_total:,}</b> in total sales — averaging roughly €{int(baseline):,} on a normal day."
    ]

    if open_days.empty:
        return " ".join(lines)

    shap = get_shap_explanations(store_id)
    driver = top_driver_phrase(shap)
    next_trading_date = open_days.iloc[0]["Date"]

    best = open_days.loc[open_days["PredictedSales"].idxmax()]
    worst = open_days.loc[open_days["PredictedSales"].idxmin()]
    best_pct = (best["PredictedSales"] / baseline - 1) * 100 if baseline else 0
    worst_pct = (worst["PredictedSales"] / baseline - 1) * 100 if baseline else 0

    if best["Date"] == worst["Date"]:
        lines.append(
            f"Only <b>{best['Date'].strftime('%A, %b %d')}</b> is open this week, so that single day "
            f"carries the whole forecast."
        )
    else:
        best_reason = _day_reason(best["Date"], best_pct, calendar, next_trading_date, driver)
        lines.append(
            f"<b>When to expect more:</b> sales look strongest on <b>{best['Date'].strftime('%A, %b %d')}</b> — "
            f"around <b>€{int(best['PredictedSales']):,}</b>, {_magnitude_word(best_pct)} above this store's "
            f"usual pace (<b>{best_pct:+.0f}%</b>), mainly thanks to {best_reason}."
        )
        worst_reason = _day_reason(worst["Date"], worst_pct, calendar, next_trading_date, driver)
        if worst_reason == best_reason:
            worst_reason = "this store's normal day-to-day dip, nothing to act on"
        lines.append(
            f"<b>{worst['Date'].strftime('%A, %b %d')}</b> looks like the quietest day — around "
            f"<b>€{int(worst['PredictedSales']):,}</b> ({_magnitude_word(worst_pct)} below normal, "
            f"<b>{worst_pct:+.0f}%</b>) — likely just {worst_reason}."
        )

    closed_days = forecast[forecast["PredictedSales"] == 0]
    if not closed_days.empty:
        names = ", ".join(d.strftime("%A, %b %d") for d in closed_days["Date"])
        lines.append(
            f"Also worth noting: the store is <b>closed</b> on {names}, so €0 is expected "
            f"{'that day' if len(closed_days) == 1 else 'those days'} — not a forecasting gap, just no trading."
        )

    return " ".join(lines)


def render_kpi_row(store_id: int, forecast, calendar):
    open_days = forecast[forecast["PredictedSales"] > 0]
    week_total = int(open_days["PredictedSales"].sum())
    avg_daily = week_total / len(open_days) if len(open_days) else 0
    baseline = store_baseline(store_id)
    pct_vs_baseline = (avg_daily / baseline - 1) * 100 if baseline else 0
    promo_days = int(calendar.loc[calendar["Open"] == 1, "Promo"].sum()) if not calendar.empty else 0
    closed_days = 7 - len(open_days)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Next 7 Days", f"€{week_total:,}")
    c2.metric("📊 Daily Average", f"€{avg_daily:,.0f}", delta=f"{pct_vs_baseline:+.0f}% vs usual")
    c3.metric("🏷️ Promo Days", f"{promo_days} of 7")
    c4.metric("🚪 Closed Days", f"{closed_days} of 7")


def build_shap_driver_chart(shap_dict: dict):
    if not shap_dict:
        return None
    items = list(shap_dict.items())[::-1]
    labels = [FEATURE_PHRASES.get(name, name).capitalize() for name, _ in items]
    values = [info["value"] for _, info in items]
    colors = ["#43D9A4" if v > 0 else "#FF6584" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.2f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Top Drivers — Tomorrow's Prediction", font=dict(size=18, color="#E0E0FF")),
        xaxis_title="Impact (SHAP value)", font=dict(size=13, color="#8B8FA8"),
        template="plotly_dark", height=340, margin=dict(t=40, l=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


def build_shap_waterfall_chart(waterfall_data: dict, top_n: int = 6):
    if not waterfall_data:
        return None
    base_sales = waterfall_data["base_value_sales"]
    pred_sales = waterfall_data["predicted_sales"]
    features = waterfall_data["features"]

    running_log = np.log1p(base_sales)
    running_dollar = base_sales
    names, deltas = [], []
    for feat in features[:top_n]:
        running_log += feat["shap_value"]
        new_dollar = np.expm1(running_log)
        names.append(FEATURE_PHRASES.get(feat["name"], feat["name"]).capitalize())
        deltas.append(new_dollar - running_dollar)
        running_dollar = new_dollar

    if len(features) > top_n:
        running_log += sum(f["shap_value"] for f in features[top_n:])
        new_dollar = np.expm1(running_log)
        names.append("All other factors")
        deltas.append(new_dollar - running_dollar)

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(names) + ["total"],
        x=["Typical day"] + names + ["Tomorrow"],
        y=[base_sales] + deltas + [pred_sales],
        text=[f"€{base_sales:,.0f}"] + [f"{'+' if d >= 0 else ''}€{d:,.0f}" for d in deltas] + [f"€{pred_sales:,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color="#2E3250")),
        increasing=dict(marker=dict(color="#43D9A4")),
        decreasing=dict(marker=dict(color="#FF6584")),
        totals=dict(marker=dict(color="#6C63FF")),
    ))
    fig.update_layout(
        title=dict(text=f"How We Got Tomorrow's Number ({waterfall_data['date']})", font=dict(size=18, color="#E0E0FF")),
        yaxis_title="Sales (€)", font=dict(size=13, color="#8B8FA8"),
        template="plotly_dark", height=400, margin=dict(t=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


def build_promo_whatif_chart(store_id: int):
    promo_on = get_whatif_forecast(store_id, True)
    promo_off = get_whatif_forecast(store_id, False)
    if promo_on.empty or promo_off.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=promo_off["Date"], y=promo_off["PredictedSales"], name="Without Promo", marker_color="#6b7280"))
    fig.add_trace(go.Bar(x=promo_on["Date"], y=promo_on["PredictedSales"], name="With Promo", marker_color="#43D9A4"))
    fig.update_layout(
        title=dict(text="What-If Scenario: Promo On vs Off", font=dict(size=18, color="#E0E0FF")),
        barmode="group", xaxis_title="Date", yaxis_title="Sales (€)",
        font=dict(size=13, color="#8B8FA8"), legend=dict(font=dict(size=12)),
        template="plotly_dark", height=380, margin=dict(t=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_model_comparison_chart(store_id: int):
    comp = get_baseline_comparison(store_id)
    if comp.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["Baseline_MovingAvg"], mode="lines+markers",
                              name="Naive Baseline (7-day avg)", line=dict(color="#8B8FA8", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["XGBoost"], mode="lines+markers",
                              name="XGBoost", line=dict(color="#FFA552", width=2)))
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["LightGBM"], mode="lines+markers",
                              name="LightGBM (primary)", line=dict(color="#6C63FF", width=3)))
    fig.update_layout(
        title=dict(text="Model Agreement — Is This Forecast Trustworthy?", font=dict(size=18, color="#E0E0FF")),
        xaxis_title="Date", yaxis_title="Sales (€)", font=dict(size=13, color="#8B8FA8"),
        legend=dict(font=dict(size=12)),
        hovermode="x unified", template="plotly_dark", height=380, margin=dict(t=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── State Machine ─────────────────────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "greeting"
    st.session_state.store_id = None

if st.session_state.stage == "greeting":
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero-card">
                <h3 style="color: #6C63FF; margin-top:0; font-size: 1.5rem;">👋 Sales Forecasting Assistant</h3>
                <p style="color: #A0A5C0; font-size: 1.05rem; margin-bottom: 1.2rem;">
                    Select a Store ID below to generate a 7-day recursive LightGBM sales forecast with confidence ranges, driver explanations, and promo scenario simulations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("store_form", clear_on_submit=False):
            store_id_val = st.number_input("Select Store ID (1–1115)", min_value=1, max_value=1115, value=1, step=1)
            submitted = st.form_submit_button("🚀 Generate 7-Day Forecast", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="feature-card">
                <span style="font-size: 2rem;">📈</span>
                <div>
                    <strong style="color: #E0E0FF; font-size: 1rem;">Multi-Step Recursive Engine</strong>
                    <div style="color: #8B8FA8; font-size: 0.85rem;">LightGBM & XGBoost with lag-feature updating</div>
                </div>
            </div>
            <div class="feature-card">
                <span style="font-size: 2rem;">🔬</span>
                <div>
                    <strong style="color: #E0E0FF; font-size: 1rem;">SHAP Driver Analysis</strong>
                    <div style="color: #8B8FA8; font-size: 0.85rem;">Deconstruct day-to-day drivers in € & log-units</div>
                </div>
            </div>
            <div class="feature-card">
                <span style="font-size: 2rem;">⚡</span>
                <div>
                    <strong style="color: #E0E0FF; font-size: 1rem;">What-If Scenario Simulation</strong>
                    <div style="color: #8B8FA8; font-size: 0.85rem;">Simulate sales with promo forced ON vs OFF</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        st.session_state.store_id = int(store_id_val)
        st.session_state.stage = "revealing"
        st.rerun()

elif st.session_state.stage == "revealing":
    store_id = st.session_state.store_id
    forecast = get_7day_forecast(store_id)

    if forecast.empty:
        st.session_state.stage = "no_data"
        st.rerun()

    is_closed = forecast["PredictedSales"] == 0
    point_labels = [
        f"{'Closed — €0' if c else f'€{v:,.0f}'}<br>{d.strftime('%A, %b %d')}"
        for c, v, d in zip(is_closed, forecast["PredictedSales"], forecast["Date"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["UpperBound"],
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["LowerBound"],
                              fill="tonexty", fillcolor="rgba(67,217,164,0.12)",
                              line=dict(width=0), name="Confidence range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=forecast["PredictedSales"],
        mode="lines+markers", name="Predicted Sales",
        line=dict(color="#43D9A4", width=3),
        marker=dict(
            size=9,
            color=["#6b7280" if c else "#43D9A4" for c in is_closed],
            symbol=["x" if c else "circle" for c in is_closed],
        ),
        text=point_labels, hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=f"Store {store_id} — 7-Day Sales Forecast", font=dict(size=20, color="#E0E0FF")),
                       xaxis_title="Date", yaxis_title="Sales (€)", font=dict(size=14, color="#8B8FA8"),
                       legend=dict(font=dict(size=12)),
                       hovermode="x unified", template="plotly_dark", height=400,
                       margin=dict(t=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{store_id}")
    if is_closed.any():
        st.caption("✕ marks days the store is closed — €0 is expected, not a prediction gap.")

    time.sleep(1.5)
    st.session_state.stage = "explaining"
    st.rerun()

elif st.session_state.stage == "no_data":
    store_id = st.session_state.store_id
    info = get_missing_store_info(store_id)

    if info is None:
        st.markdown(f"Store **{store_id}** has no sales history at all in the database.")
    else:
        window_dates = pd.Series(info["dates"])
        flat = pd.DataFrame({"Date": window_dates, "PredictedSales": 0})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flat["Date"], y=flat["PredictedSales"], mode="lines+markers",
            name="No forecast available", line=dict(color="#6b7280", width=3, dash="dash"),
            marker=dict(size=9, color="#6b7280", symbol="x"),
            text=[d.strftime("%A, %b %d") for d in flat["Date"]],
            hovertemplate="No forecast — %{text}<extra></extra>",
        ))
        fig.update_layout(title=dict(text=f"Store {store_id} — No Forecast Available", font=dict(size=20, color="#E0E0FF")),
                           xaxis_title="Date", font=dict(size=14, color="#8B8FA8"),
                           yaxis_title="Sales (€)", yaxis=dict(range=[-1, 10]), template="plotly_dark",
                           height=400, margin=dict(t=40), paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_nodata_{store_id}")

        st.warning(
            f"The line shows **€0 because we have nothing to predict from** — not because Store "
            f"{store_id} is expected to sell nothing. Its own recent trading days average around "
            f"**€{info['recent_avg']:,.0f}**, a normal active store. The source dataset `test.csv` "
            f"did not include a calendar window for Store {store_id}."
        )

    if st.button("← Try another store"):
        st.session_state.stage = "greeting"
        st.rerun()

elif st.session_state.stage == "explaining":
    store_id = st.session_state.store_id
    forecast = get_7day_forecast(store_id)
    calendar = get_forecast_calendar(store_id)
    narrative = build_narrative(store_id, forecast, calendar)

    render_kpi_row(store_id, forecast, calendar)
    st.divider()

    st.markdown("### 🔬 Why This Forecast")
    shap_dict = get_shap_explanations(store_id)
    waterfall_data = get_shap_waterfall_data(store_id)
    col1, col2 = st.columns(2)
    with col1:
        fig = build_shap_driver_chart(shap_dict)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"shap_bar_{store_id}")
        else:
            st.info("No SHAP drivers available for this store.")
    with col2:
        fig = build_shap_waterfall_chart(waterfall_data)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"shap_wf_{store_id}")
        else:
            st.info("No waterfall breakdown available for this store.")

    st.divider()
    st.markdown("### ⚡ What-If: Promotion Impact")
    fig = build_promo_whatif_chart(store_id)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"promo_{store_id}")

    st.divider()
    st.markdown("### 🤖 Model Agreement & Reliability")
    fig = build_model_comparison_chart(store_id)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"modelcmp_{store_id}")

    st.divider()
    st.markdown(
        f"""
        <div class="hero-card">
            <h4 style="color: #43D9A4; margin-top:0; font-size: 1.3rem;">📋 Executive Summary — Store {store_id}</h4>
            <div style="color: #E0E0FF; font-size: 1.05rem; line-height: 1.7;">{narrative}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── CSV Export ─────────────────────────────────────────────────────────────
    csv_bytes = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export 7-Day Forecast as CSV",
        data=csv_bytes,
        file_name=f"store_{store_id}_7day_forecast.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Explore Another Store", use_container_width=True):
            st.session_state.stage = "greeting"
            st.session_state.store_id = None
            st.rerun()
    with c2:
        if st.button("✅ Finish Exploration", use_container_width=True):
            st.session_state.stage = "done"
            st.rerun()

elif st.session_state.stage == "done":
    store_id = st.session_state.store_id
    forecast = get_7day_forecast(store_id)
    calendar = get_forecast_calendar(store_id)
    narrative = build_narrative(store_id, forecast, calendar)

    st.markdown(
        f"""
        <div class="hero-card">
            <h4 style="color: #43D9A4; margin-top:0; font-size: 1.3rem;">📋 Executive Summary — Store {store_id}</h4>
            <div style="color: #E0E0FF; font-size: 1.05rem; line-height: 1.7;">{narrative}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.success("Analysis complete! Click below if you wish to analyze another store.")
    if st.button("← Start New Forecast"):
        st.session_state.stage = "greeting"
        st.session_state.store_id = None
        st.rerun()

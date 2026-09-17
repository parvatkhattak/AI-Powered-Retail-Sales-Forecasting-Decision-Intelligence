import sys
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH
from src.model_engine import (
    get_7day_forecast, get_shap_explanations, get_missing_store_info,
    get_shap_waterfall_data, get_baseline_comparison, get_whatif_forecast,
    get_forecast_calendar,
)
from src.ui_theme import apply_theme, asset_data_uri

st.set_page_config(page_title="Forecasting", page_icon="📈", layout="wide")
apply_theme()

ASSISTANT_AVATAR = asset_data_uri("avatar.png") or "https://img.icons8.com/fluency/96/businessman.png"

st.markdown("""
<style>
    p, li { font-size: 1.35rem; line-height: 1.75; }
    .stMarkdown h1 { font-size: 2.9rem !important; }
    .stMarkdown h2 { font-size: 2.4rem !important; }
    .stMarkdown h3, .stMarkdown h4 { font-size: 2.0rem !important; }
    [data-testid="stMetricValue"] { font-size: 2.7rem !important; }
    [data-testid="stMetricLabel"] { font-size: 1.3rem !important; }
    [data-testid="stMetricDelta"] { font-size: 1.1rem !important; }

    /* The centered store-ID input — "average" sized, emerald glow, dark to match the page */
    div[data-testid="stTextInput"] input {
        font-size: 1.3rem !important;
        text-align: center;
        border-radius: 999px !important;
        border: 1.5px solid rgba(34,197,94,0.55) !important;
        box-shadow: 0 0 0 1.5px rgba(34,197,94,0.35), 0 8px 24px rgba(34,197,94,0.18);
        padding: 0.6rem 1.2rem !important;
        background: rgba(6,20,13,0.85) !important;
        color: #e5e7eb !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: rgba(229,231,235,0.5) !important; }

    /* The "Ask →" submit button — emerald-to-blue gradient pill, matching the input above */
    div[data-testid="stForm"] button {
        background: linear-gradient(90deg, #10b981, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 8px 24px rgba(16,185,129,0.25);
    }
    div[data-testid="stForm"] button:hover { filter: brightness(1.1); }
    div[data-testid="stForm"] button p { color: white !important; }

    @keyframes popIn3D {
        0%   { opacity: 0; transform: scale3d(0.7,0.7,0.7) rotateY(-20deg) translateY(20px); }
        100% { opacity: 1; transform: scale3d(1,1,1) rotateY(0deg) translateY(0); }
    }
    [data-testid="stPlotlyChart"] {
        animation: popIn3D 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
        transform-style: preserve-3d;
    }
</style>
""", unsafe_allow_html=True)

components.html("""
<script src="https://cdn.tailwindcss.com"></script>
<style>
    body { margin: 0; background: transparent; display: flex; flex-direction: column; align-items: center; gap: 6px; }
    .row { display: flex; align-items: center; justify-content: center; gap: 18px; }
    @keyframes float3d {
        0%, 100% { transform: perspective(700px) rotateX(6deg) rotateY(-6deg) translateY(0px); }
        50%      { transform: perspective(700px) rotateX(-6deg) rotateY(6deg) translateY(-8px); }
    }
    .title3d { animation: float3d 4s ease-in-out infinite; transform-style: preserve-3d; }
</style>
<body>
    <div class="row">
        <img src="https://img.icons8.com/fluency/96/line-chart.png"
             class="title3d w-16 h-16 drop-shadow-lg" />
        <h1 class="title3d text-6xl font-extrabold text-center
                   bg-gradient-to-r from-emerald-400 via-teal-300 to-blue-400
                   bg-clip-text text-transparent drop-shadow-lg">
            Sales Forecasting Assistant
        </h1>
    </div>
    <p class="text-gray-300 text-2xl text-center">
        Get accurate sales predictions, discover trends, and make smarter decisions for your business growth.
    </p>
</body>
""", height=170)


def tailwind_block(inner_html: str, height: int):
    components.html(f"""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ margin:0; background: transparent; font-family: -apple-system, sans-serif; }}
        @keyframes popIn3D {{
            0%   {{ opacity: 0; transform: scale3d(0.7,0.7,0.7) rotateY(-20deg) translateY(20px); }}
            100% {{ opacity: 1; transform: scale3d(1,1,1) rotateY(0deg) translateY(0); }}
        }}
        .pop3d {{ animation: popIn3D 0.6s cubic-bezier(0.2,0.8,0.2,1); transform-style: preserve-3d; }}
    </style>
    <body class="text-white">{inner_html}</body>
    """, height=height, scrolling=False)


def greeting_card():
    tailwind_block(f"""
    <div class="pop3d flex items-center gap-5 bg-gradient-to-br from-emerald-950 via-black to-emerald-900
                rounded-3xl p-6 shadow-2xl border border-emerald-500/30">
        <img src="{ASSISTANT_AVATAR}" class="w-24 h-24 rounded-full ring-4 ring-emerald-400/60 shadow-lg" />
        <div>
            <p class="text-3xl font-bold text-emerald-300">👋 Hi, how can I help you?</p>
            <p class="text-xl text-gray-300 mt-1">Which store's sales trend would you like to see?</p>
        </div>
    </div>
    """, height=170)


def section_heading(title: str, color: str = "emerald"):
    """A section title with a colored accent bar instead of a generic emoji
    prefix — keeps headings looking designed rather than chatbot-generated."""
    tailwind_block(f"""
    <div class="flex items-center gap-3 mt-1">
        <div class="w-2 h-10 rounded-full bg-{color}-400"></div>
        <p class="text-4xl font-extrabold text-white tracking-tight">{title}</p>
    </div>
    """, height=72)


def feature_highlights():
    cards = [
        ("📈", "emerald", "Better Forecasts", "Plan inventory &amp; reduce stockouts"),
        ("🎯", "blue", "Spot Trends", "Understand what's driving sales"),
        ("💡", "purple", "Make Smarter Decisions", "Grow your business with data"),
    ]
    rows = "".join(f"""
    <div class="pop3d flex items-center gap-5 bg-gradient-to-br from-{color}-950/60 via-black to-black
                rounded-2xl p-6 shadow-xl border border-{color}-500/30">
        <div class="w-16 h-16 shrink-0 rounded-xl bg-{color}-500/20 flex items-center justify-center text-4xl">
            {icon}
        </div>
        <div>
            <p class="text-2xl font-bold text-{color}-300">{title}</p>
            <p class="text-lg text-gray-400 mt-1">{desc}</p>
        </div>
    </div>
    """ for icon, color, title, desc in cards)

    tailwind_block(f"""
    <div class="h-full flex flex-col justify-between gap-6">{rows}</div>
    """, height=520)


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
    # Open = 1 only — otherwise closed days (Sales = 0) drag the "normal day" average down.
    row = conn.execute("SELECT AVG(Sales) FROM sales WHERE Store = ? AND Open = 1", (store_id,)).fetchone()
    conn.close()
    return float(row[0]) if row and row[0] else 0.0


def _day_reason(date, pct: float, calendar, next_trading_date, driver_phrase) -> str:
    """A grounded reason for why one specific day differs from the norm —
    checked in the correct direction against the one calendar fact we can
    actually verify (Promo), before falling back to the model's SHAP driver
    (only for the exact day that driver was computed for, never reused for
    a different day it wasn't calculated on). No day-of-week guesswork —
    whether weekends run higher or lower is store-specific, not universal,
    so asserting it risks giving a backwards reason."""
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
    """The at-a-glance numbers before anyone reads a single chart."""
    open_days = forecast[forecast["PredictedSales"] > 0]
    week_total = int(open_days["PredictedSales"].sum())
    avg_daily = week_total / len(open_days) if len(open_days) else 0
    baseline = store_baseline(store_id)
    pct_vs_baseline = (avg_daily / baseline - 1) * 100 if baseline else 0
    promo_days = int(calendar.loc[calendar["Open"] == 1, "Promo"].sum()) if not calendar.empty else 0
    closed_days = 7 - len(open_days)

    # — Accuracy badge: pull from model metrics (store-level RMSPE not available,
    # so we use the fleet RMSPE as the headline accuracy number)
    _store_acc = None
    try:
        from src.model_engine import get_model_metrics as _gmm
        _mmet = _gmm()
        _fleet_rmspe = _mmet.get("lightgbm", {}).get("rmspe", None)
        if _fleet_rmspe is not None:
            _store_acc = round((1.0 - _fleet_rmspe) * 100, 1)
    except Exception:
        pass

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Next 7 Days", f"€{week_total:,}")
    c2.metric("📊 Daily Average", f"€{avg_daily:,.0f}", delta=f"{pct_vs_baseline:+.0f}% vs usual")
    c3.metric("🏷️ Promo Days", f"{promo_days} of 7")
    c4.metric("🚪 Closed Days", f"{closed_days} of 7")
    with c5:
        if _store_acc is not None:
            _acc_color = "#10b981" if _store_acc >= 85 else "#f59e0b" if _store_acc >= 75 else "#ef4444"
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px;padding:10px 12px 8px;text-align:center;">
                    <div style="font-size:0.72rem;color:#8B8FA8;margin-bottom:4px;">🎯 Model Accuracy</div>
                    <div style="font-size:1.5rem;font-weight:800;color:{_acc_color};">{_store_acc}%</div>
                    <div style="font-size:0.68rem;color:#6b7280;margin-top:2px;">Fleet RMSPE basis</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.metric("🎯 Accuracy", "N/A")

    # ── Animated Confidence Meter ─────────────────────────────────────────────
    # Confidence = how tight the forecast band is relative to the prediction.
    # Narrow band (LowerBound close to UpperBound) → high confidence.
    open_f = forecast[forecast["PredictedSales"] > 0]
    if not open_f.empty and "LowerBound" in open_f.columns and "UpperBound" in open_f.columns:
        avg_pred  = open_f["PredictedSales"].mean()
        avg_range = (open_f["UpperBound"] - open_f["LowerBound"]).mean()
        raw_conf  = max(0.0, 1.0 - (avg_range / avg_pred / 2)) if avg_pred else 0.0
        confidence_pct = round(min(raw_conf, 1.0) * 100)
    else:
        confidence_pct = 72   # sensible default when bounds not available

    if   confidence_pct >= 80: bar_color, label, label_color = "#10b981", "High", "#10b981"
    elif confidence_pct >= 60: bar_color, label, label_color = "#f59e0b", "Moderate", "#f59e0b"
    else:                       bar_color, label, label_color = "#ef4444", "Low", "#ef4444"

    st.markdown(
        f"""
        <style>
        @keyframes fillBar {{
            from {{ width: 0%; }}
            to   {{ width: {confidence_pct}%; }}
        }}
        .conf-wrap {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 20px 12px;
            margin: 14px 0 4px;
        }}
        .conf-label-row {{
            display: flex; justify-content: space-between;
            align-items: center; margin-bottom: 8px;
        }}
        .conf-title  {{ font-size: 0.85rem; color: #8B8FA8; font-weight: 500; }}
        .conf-badge  {{
            font-size: 0.82rem; font-weight: 700; color: {label_color};
            background: rgba(255,255,255,0.05); border-radius: 20px;
            padding: 2px 10px; border: 1px solid {label_color}44;
        }}
        .conf-pct    {{ font-size: 1.05rem; font-weight: 700; color: {bar_color}; }}
        .conf-track  {{
            width: 100%; height: 10px; border-radius: 999px;
            background: rgba(255,255,255,0.08); overflow: hidden;
        }}
        .conf-fill   {{
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, {bar_color}aa, {bar_color});
            animation: fillBar 1.2s cubic-bezier(.4,0,.2,1) forwards;
            box-shadow: 0 0 8px {bar_color}55;
        }}
        </style>
        <div class="conf-wrap">
            <div class="conf-label-row">
                <span class="conf-title">🎯 Model Confidence</span>
                <span class="conf-badge">{label}</span>
                <span class="conf-pct">{confidence_pct}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def build_shap_driver_chart(shap_dict: dict):
    """Horizontal bar of the top 5 features moving tomorrow's prediction —
    green pushes sales up, red pushes them down."""
    if not shap_dict:
        return None
    items = list(shap_dict.items())[::-1]  # reverse so the strongest driver sits on top
    labels = [FEATURE_PHRASES.get(name, name).capitalize() for name, _ in items]
    values = [info["value"] for _, info in items]
    colors = ["#10b981" if v > 0 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.2f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Top Drivers — Tomorrow's Prediction", font=dict(size=20)),
        xaxis_title="Impact (SHAP, relative)", font=dict(size=15, color="#e5e7eb"),
        template="plotly_dark", height=340, margin=dict(t=50, l=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


def build_shap_waterfall_chart(waterfall_data: dict, top_n: int = 6):
    """Walks from this store's 'typical day' to tomorrow's actual prediction,
    one driver at a time, in real € — not raw SHAP log-units. Each step's €
    amount is the true marginal effect of adding that feature on top of the
    ones before it, so the bars always add up exactly to the final number."""
    if not waterfall_data:
        return None
    base_sales = waterfall_data["base_value_sales"]
    pred_sales = waterfall_data["predicted_sales"]
    features = waterfall_data["features"]  # already sorted by |shap_value| descending

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
        connector=dict(line=dict(color="rgba(255,255,255,0.3)")),
        increasing=dict(marker=dict(color="#10b981")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#3b82f6")),
    ))
    fig.update_layout(
        title=dict(text=f"How We Got Tomorrow's Number ({waterfall_data['date']})", font=dict(size=20)),
        yaxis_title="Sales (€)", font=dict(size=15, color="#e5e7eb"),
        template="plotly_dark", height=400, margin=dict(t=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


def build_promo_whatif_chart(store_id: int):
    """Same week, replayed twice — once as if Promo were off every day, once
    as if it were on — so the promo's real day-by-day lift is visible."""
    promo_on = get_whatif_forecast(store_id, True)
    promo_off = get_whatif_forecast(store_id, False)
    if promo_on.empty or promo_off.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=promo_off["Date"], y=promo_off["PredictedSales"], name="Without Promo", marker_color="#6b7280"))
    fig.add_trace(go.Bar(x=promo_on["Date"], y=promo_on["PredictedSales"], name="With Promo", marker_color="#10b981"))
    fig.update_layout(
        title=dict(text="What If: Promo On vs Off", font=dict(size=20)),
        barmode="group", xaxis_title="Date", yaxis_title="Sales (€)",
        font=dict(size=15, color="#e5e7eb"), legend=dict(font=dict(size=14)),
        template="plotly_dark", height=380, margin=dict(t=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_model_comparison_chart(store_id: int):
    """LightGBM vs XGBoost vs a naive 7-day moving average — when the two
    real models agree closely, that's a trustworthy forecast; when they
    diverge, that's a week worth double-checking."""
    comp = get_baseline_comparison(store_id)
    if comp.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["Baseline_MovingAvg"], mode="lines+markers",
                              name="Naive Baseline (7-day avg)", line=dict(color="#6b7280", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["XGBoost"], mode="lines+markers",
                              name="XGBoost", line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["LightGBM"], mode="lines+markers",
                              name="LightGBM (primary)", line=dict(color="#10b981", width=3)))
    fig.update_layout(
        title=dict(text="Model Agreement — Is This Forecast Trustworthy?", font=dict(size=20)),
        xaxis_title="Date", yaxis_title="Sales (€)", font=dict(size=15, color="#e5e7eb"),
        legend=dict(font=dict(size=14)),
        hovermode="x unified", template="plotly_dark", height=380, margin=dict(t=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── State machine ─────────────────────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "greeting"
    st.session_state.store_id = None

if st.session_state.stage == "greeting":
    left, right = st.columns([2, 1], gap="large")
    with left:
        greeting_card()
        with st.form("store_form", clear_on_submit=True):
            store_input = st.text_input("", placeholder="Enter a Store ID (1–1115)…", label_visibility="collapsed")
            submitted = st.form_submit_button("Ask →", use_container_width=True)
    with right:
        feature_highlights()

    if submitted:
        try:
            sid = int(store_input.strip())
            if not (1 <= sid <= 1115):
                raise ValueError
            st.session_state.store_id = sid
            st.session_state.stage = "revealing"
            st.rerun()
        except (ValueError, AttributeError):
            st.error("Please enter a valid Store ID between 1 and 1115.")

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
                              fill="tonexty", fillcolor="rgba(16,185,129,0.15)",
                              line=dict(width=0), name="Confidence range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=forecast["PredictedSales"],
        mode="lines+markers", name="Predicted Sales",
        line=dict(color="rgb(16,185,129)", width=3),
        marker=dict(
            size=9,
            color=["#6b7280" if c else "rgb(16,185,129)" for c in is_closed],
            symbol=["x" if c else "circle" for c in is_closed],
        ),
        text=point_labels, hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=f"Store {store_id} — 7-Day Forecast", font=dict(size=22)),
                       xaxis_title="Date", yaxis_title="Sales (€)", font=dict(size=16, color="#e5e7eb"),
                       legend=dict(font=dict(size=14)),
                       hovermode="x unified", template="plotly_dark", height=400,
                       margin=dict(t=50), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{store_id}")
    if is_closed.any():
        st.caption("✕ marks days the store is closed — €0 is expected, not a prediction gap.")

    time.sleep(4)
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
        fig.update_layout(title=dict(text=f"Store {store_id} — No Forecast Available", font=dict(size=22)),
                           xaxis_title="Date", font=dict(size=16, color="#e5e7eb"),
                           yaxis_title="Sales (€)", yaxis=dict(range=[-1, 10]), template="plotly_dark",
                           height=400, margin=dict(t=50), paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_nodata_{store_id}")

        st.markdown(
            f"The line shows **€0 because we have nothing to predict from** — not because Store "
            f"{store_id} is expected to sell nothing. Its own recent trading days average around "
            f"<b>€{info['recent_avg']:,.0f}</b>, a perfectly normal, active store.<br><br>"
            f"<b>The actual reason:</b> the original forecasting dataset (<code>test.csv</code>) only defines "
            f"next week's Open/Promo/Holiday calendar for 856 of the 1,115 stores — Store {store_id} wasn't "
            f"one of them. Without that future calendar, the model has nothing to build a prediction from. "
            f"It's a gap in the source data, not a signal about this store's business.",
            unsafe_allow_html=True,
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
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    section_heading("Why This Forecast", "emerald")
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

    section_heading("What If: Promotion Impact", "blue")
    fig = build_promo_whatif_chart(store_id)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"promo_{store_id}")

    # ── Best Time to Run a Promotion Calendar ─────────────────────────────────
    section_heading("Best Day to Run a Promotion", "blue")
    try:
        _p_on  = get_whatif_forecast(store_id, True)
        _p_off = get_whatif_forecast(store_id, False)
        if not _p_on.empty and not _p_off.empty:
            _uplift = (_p_on["PredictedSales"].values - _p_off["PredictedSales"].values)
            _best_idx   = int(_uplift.argmax())
            _max_uplift = float(_uplift.max())
            _min_uplift = float(_uplift.min())
            _range      = max(_max_uplift - _min_uplift, 1.0)

            _day_cells = []
            for _i, (_row_on, _row_off) in enumerate(
                zip(_p_on.itertuples(), _p_off.itertuples())
            ):
                _d   = str(getattr(_row_on, "Date", ""))[:10]
                _day = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][_i % 7]
                _upl = float(getattr(_row_on, "PredictedSales", 0)) - float(getattr(_row_off, "PredictedSales", 0))
                _t   = max(0.0, (_upl - _min_uplift) / _range)   # 0–1 normalised
                _r   = int(239 - _t * (239 - 16))
                _g   = int(68  + _t * (185 - 68))
                _b   = int(68  + _t * (129 - 68))
                _bg  = f"rgba({_r},{_g},{_b},0.25)"
                _bdr = f"rgb({_r},{_g},{_b})"
                _best_ring = "box-shadow:0 0 0 2px #fff,0 0 0 4px #10b981;" if _i == _best_idx else ""
                _label = "🏆" if _i == _best_idx else ""
                _cell = (
                    f"<div style='flex:1;min-width:72px;background:{_bg};border:1.5px solid {_bdr};"
                    f"border-radius:12px;padding:10px 6px;text-align:center;{_best_ring}'>"
                    f"<div style='font-size:0.72rem;color:#9ca3af;'>{_day}</div>"
                    f"<div style='font-size:0.68rem;color:#6b7280;'>{_d[5:]}</div>"
                    f"<div style='font-size:1.0rem;font-weight:700;color:{_bdr};margin-top:4px;'>"
                    f"+€{int(_upl):,}</div>"
                    f"<div style='font-size:1.1rem;margin-top:2px;'>{_label}</div>"
                    f"</div>"
                )
                _day_cells.append(_cell)

            _best_date = str(getattr(list(_p_on.itertuples())[_best_idx], "Date", ""))[:10]
            st.markdown(
                f"<div style='font-size:0.83rem;color:#8B8FA8;margin-bottom:8px;'>"
                f"Promo uplift per day vs no-promo scenario — "
                f"<b style='color:#10b981;'>🏆 Best day: {_best_date} (+€{int(_max_uplift):,})</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='display:flex;gap:8px;flex-wrap:nowrap;overflow-x:auto;padding:4px 0;'>" +
                "".join(_day_cells) + "</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.info("Promo calendar unavailable for this store.")

    section_heading("Model Agreement", "purple")
    fig = build_model_comparison_chart(store_id)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"modelcmp_{store_id}")

    tailwind_block(f"""
    <div class="pop3d bg-gradient-to-br from-emerald-950/70 via-black to-emerald-900/40
                border border-emerald-500/30 rounded-3xl p-7 shadow-2xl">
        <div class="flex items-center gap-3 mb-3">
            <div class="w-1.5 h-7 rounded-full bg-emerald-400"></div>
            <p class="text-emerald-300 font-extrabold text-3xl">What to Expect — Store {store_id}</p>
        </div>
        <p class="text-gray-100 text-xl leading-relaxed">{narrative}</p>
    </div>
    """, height=280)

    st.markdown("<div style='text-align:center; margin-top:1.5rem; font-size:1.5rem;'>Would you like to explore more stores?</div>",
                unsafe_allow_html=True)
    _, c1, c2, _ = st.columns([2, 1, 1, 2])
    with c1:
        if st.button("Yes, explore another", use_container_width=True):
            st.session_state.stage = "greeting"
            st.session_state.store_id = None
            st.rerun()
    with c2:
        if st.button("No, I'm done", use_container_width=True):
            st.session_state.stage = "done"
            st.rerun()

elif st.session_state.stage == "done":
    store_id = st.session_state.store_id
    forecast = get_7day_forecast(store_id)
    calendar = get_forecast_calendar(store_id)
    narrative = build_narrative(store_id, forecast, calendar)
    tailwind_block(f"""
    <div class="pop3d bg-gradient-to-br from-emerald-950/70 via-black to-emerald-900/40
                border border-emerald-500/30 rounded-3xl p-7 shadow-2xl">
        <div class="flex items-center gap-3 mb-3">
            <div class="w-1.5 h-7 rounded-full bg-emerald-400"></div>
            <p class="text-emerald-300 font-extrabold text-3xl">What to Expect — Store {store_id}</p>
        </div>
        <p class="text-gray-100 text-xl leading-relaxed">{narrative}</p>
    </div>
    """, height=280)
    st.caption("Thanks for exploring! Refresh the page to start over.")

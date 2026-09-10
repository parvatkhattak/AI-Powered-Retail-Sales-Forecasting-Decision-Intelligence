"""
pages/5_⚙️_Model_Performance.py
Owner: Dikshit — UI Developer

ML transparency and model evaluation page.

Sections:
  1. Model KPI Cards — RMSPE, MAE, R² for LightGBM, XGBoost, and Baseline
  2. Model Comparison Chart — grouped bar: RMSPE + R² side by side
  3. Walk-Forward CV Diagram — visual illustration of the time-series validation structure
  4. MAE Comparison Table — detailed metric table with download
  5. Training Info — dataset period, feature count, primary model used

Architecture constraints:
  - Only calls get_model_metrics() from src/model_engine.py (reads models/metrics.json — no .pkl needed)
  - All constants from config.py
  - All errors handled gracefully
"""

import sys
from pathlib import Path
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import USE_MOCKS
from src.model_engine import get_model_metrics, METRICS_PATH
from components.charts import (
    plot_model_metrics_comparison,
    _apply_base,
    PALETTE_PRIMARY,
    PALETTE_GREEN,
    PALETTE_ORANGE,
    PALETTE_MUTED,
    PALETTE_TEXT,
    PALETTE_SURFACE,
    PALETTE_BORDER,
    PALETTE_ACCENT,
    PALETTE_RED,
)
from components.ui_helpers import (
    kpi_card,
    section_header,
    show_empty_state,
    download_csv_button,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Performance — Retail AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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
    .info-box {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2133 100%);
        border: 1px solid #2E3250; border-radius: 12px; padding: 20px 24px;
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached Fetcher ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def cached_model_metrics() -> dict:
    try:
        return get_model_metrics()
    except Exception:
        # Fallback: try reading metrics.json directly
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=52)
    st.markdown("## ⚙️ Model Performance")
    st.divider()

    mode_badge = (
        '<span class="badge badge-mock">🟡 MOCK DATA</span>'
        if USE_MOCKS
        else '<span class="badge badge-live">🟢 LIVE DATA</span>'
    )
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.divider()

    st.markdown(
        """
        **What this page shows:**
        - RMSPE / MAE / R² for each model
        - Side-by-side comparison vs baseline
        - Walk-forward CV structure diagram
        - Training data information

        **Why RMSPE?**
        It's the official Rossmann Kaggle metric.
        Penalises % error equally for high-sales
        and low-sales stores.

        **Industry threshold:** RMSPE < 0.15 is excellent.
        """
    )
    st.divider()
    st.caption("📦 Dataset: Rossmann Store Sales")
    st.caption("🏗️ Owner: Dikshit (pages/)")

# ── Page Header ───────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="page-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown("ML transparency dashboard — evaluation metrics, baseline comparison, and model training details.")
with col_badge:
    st.markdown(
        f"<div style='text-align:right;padding-top:12px;'>{mode_badge}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Load Metrics ──────────────────────────────────────────────────────────────
metrics = cached_model_metrics()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI Cards
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "📊 Evaluation Metrics",
    "RMSPE, MAE, and R² across LightGBM (primary), XGBoost, and Baseline (7-day moving average)",
)

if metrics:
    primary = metrics.get("primary_model", "lightgbm")

    # ── Row 1: RMSPE (lower is better) ──
    st.markdown("###### RMSPE — Root Mean Squared Percentage Error (lower = better, industry good = <0.15)")
    r1, r2, r3 = st.columns(3)

    lgbm = metrics.get("lightgbm", {})
    xgb  = metrics.get("xgboost", {})
    base = metrics.get("baseline", {})

    with r1:
        kpi_card(
            "🥇 LightGBM RMSPE",
            f"{lgbm.get('rmspe', 0):.4f}",
            delta="Primary Model ✅" if primary == "lightgbm" else None,
            delta_color="normal",
            help_text="LightGBM model — lower RMSPE = more accurate forecasts.",
        )
    with r2:
        kpi_card(
            "🥈 XGBoost RMSPE",
            f"{xgb.get('rmspe', 0):.4f}",
            delta="Secondary Model",
            delta_color="off",
            help_text="XGBoost model — trained for comparison.",
        )
    with r3:
        improvement = (
            (base.get("rmspe", 0) - lgbm.get("rmspe", 0)) / base.get("rmspe", 1) * 100
            if base.get("rmspe", 0) > 0 else 0
        )
        kpi_card(
            "📏 Baseline RMSPE",
            f"{base.get('rmspe', 0):.4f}",
            delta=f"ML is {improvement:.0f}% better",
            delta_color="normal",
            help_text="Naive 7-day moving average — the benchmark our ML must beat.",
        )

    st.markdown("")

    # ── Row 2: MAE and R² ──
    st.markdown("###### MAE (€) and R² Score")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("LightGBM MAE", f"€{lgbm.get('mae', 0):,.0f}", help_text="Mean Absolute Error — average euro error per prediction.")
    with c2:
        kpi_card("LightGBM R²", f"{lgbm.get('r2', 0):.4f}", help_text="R² score: 1.0 = perfect, 0.0 = no better than mean.")
    with c3:
        kpi_card("XGBoost MAE", f"€{xgb.get('mae', 0):,.0f}")
    with c4:
        kpi_card("XGBoost R²", f"{xgb.get('r2', 0):.4f}")

else:
    show_empty_state(
        "Metrics not found. Run model training first: `.venv/bin/python src/model_engine.py --train`"
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Comparison Chart
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "📈 Model Comparison",
    "Visual comparison of RMSPE and R² across all three models",
)

if metrics:
    fig_cmp = plot_model_metrics_comparison(metrics)
    st.plotly_chart(fig_cmp, use_container_width=True, key="model_cmp_chart")

    # MAE comparison as a separate bar
    model_labels = ["LightGBM", "XGBoost", "Baseline (Moving Avg)"]
    mae_values   = [
        lgbm.get("mae", 0),
        xgb.get("mae", 0),
        base.get("mae", 0),
    ]
    mae_colors   = [PALETTE_PRIMARY, PALETTE_GREEN, PALETTE_MUTED]

    fig_mae = go.Figure(go.Bar(
        x=model_labels, y=mae_values,
        marker_color=mae_colors,
        text=[f"€{v:,.0f}" for v in mae_values],
        textposition="outside",
        hovertemplate="%{x}<br>MAE: €%{y:,.0f}<extra></extra>",
    ))
    fig_mae.update_layout(
        title=dict(text="MAE (€) Comparison — lower is better", font=dict(size=16, color=PALETTE_TEXT)),
        yaxis_title="Mean Absolute Error (€)",
    )
    st.plotly_chart(_apply_base(fig_mae), use_container_width=True, key="mae_cmp_chart")
else:
    show_empty_state("Chart unavailable — train models first.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Walk-Forward CV Diagram
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "🔄 Walk-Forward Cross Validation",
    "How we validated the model without data leakage — each fold uses only past data to predict future",
)

# Build a visual Gantt-style chart representing the 5 CV folds
FOLDS = 5
TOTAL_WEEKS = 120  # ~2.5 years of weekly data
VALIDATE_WEEKS = 6  # 6 weeks (~42 days) hold-out per fold

fold_data = []
for fold in range(FOLDS):
    train_end = TOTAL_WEEKS - (FOLDS - fold) * VALIDATE_WEEKS
    val_start = train_end
    val_end   = train_end + VALIDATE_WEEKS
    fold_data.append({
        "fold": f"Fold {fold + 1}",
        "train_start": 0,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
    })

fig_cv = go.Figure()
for row in fold_data:
    # Training bar
    fig_cv.add_trace(go.Bar(
        x=[row["train_end"] - row["train_start"]],
        y=[row["fold"]],
        base=[row["train_start"]],
        orientation="h",
        name="Training Data" if row["fold"] == "Fold 1" else "",
        showlegend=(row["fold"] == "Fold 1"),
        marker_color=PALETTE_PRIMARY,
        opacity=0.7,
        hovertemplate=f"{row['fold']}<br>Train: weeks 0–{row['train_end']}<extra></extra>",
    ))
    # Validation bar
    fig_cv.add_trace(go.Bar(
        x=[row["val_end"] - row["val_start"]],
        y=[row["fold"]],
        base=[row["val_start"]],
        orientation="h",
        name="Validation (7 days)" if row["fold"] == "Fold 1" else "",
        showlegend=(row["fold"] == "Fold 1"),
        marker_color=PALETTE_ORANGE,
        opacity=0.9,
        hovertemplate=f"{row['fold']}<br>Validate: weeks {row['val_start']}–{row['val_end']}<extra></extra>",
    ))

fig_cv.update_layout(
    barmode="overlay",
    title=dict(
        text="Walk-Forward Cross Validation — 5 Folds (each = 6-week hold-out)",
        font=dict(size=16, color=PALETTE_TEXT),
    ),
    xaxis_title="Time (weeks →)",
    yaxis=dict(categoryorder="array", categoryarray=[f"Fold {i}" for i in range(FOLDS, 0, -1)]),
    legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(_apply_base(fig_cv), use_container_width=True, key="cv_diagram")

st.info(
    "**No data leakage:** In each fold, the model trains only on past data and predicts a "
    "future 6-week window it has never seen. This mirrors how the model operates in production.",
    icon="🛡️",
)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Detailed Metrics Table
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "📋 Full Metrics Table",
    "All evaluation metrics in one downloadable table",
)

if metrics:
    rows = []
    label_map = {"lightgbm": "LightGBM ⭐ Primary", "xgboost": "XGBoost", "baseline": "Baseline (7-day Moving Avg)"}
    for key in ["lightgbm", "xgboost", "baseline"]:
        if key in metrics:
            m = metrics[key]
            rows.append({
                "Model": label_map.get(key, key),
                "RMSPE": round(m.get("rmspe", 0), 6),
                "MAE (€)": round(m.get("mae", 0), 2),
                "R²": round(m.get("r2", 0), 6),
                "RMSPE Grade": (
                    "🟢 Excellent (<0.15)"  if m.get("rmspe", 1) < 0.15 else
                    "🟡 Good (0.15–0.25)"   if m.get("rmspe", 1) < 0.25 else
                    "🔴 Needs improvement"
                ),
            })

    metrics_table = pd.DataFrame(rows)
    st.dataframe(metrics_table, use_container_width=True)
    download_csv_button(metrics_table, "model_metrics.csv")
else:
    show_empty_state("Metrics table unavailable.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Training Info
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "🏗️ Training Information",
    "Details about the dataset, features, and model configuration",
)

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown(
        """
        <div class="info-box">
            <h5 style="color:#E0E0FF; margin-top:0;">📦 Dataset</h5>
            <table style="width:100%; color:#C0C0D0; font-size:0.88rem;">
                <tr><td><b>Source</b></td><td>Rossmann Store Sales (Kaggle)</td></tr>
                <tr><td><b>Stores</b></td><td>1,115 stores across Germany</td></tr>
                <tr><td><b>Date Range</b></td><td>2013-01-01 → 2015-07-31</td></tr>
                <tr><td><b>Raw Rows</b></td><td>~1,017,209 (before cleaning)</td></tr>
                <tr><td><b>After Cleaning</b></td><td>~844,338 (closed days removed)</td></tr>
                <tr><td><b>Train/Val Split</b></td><td>Last 6 weeks held out for evaluation</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info_col2:
    st.markdown(
        """
        <div class="info-box">
            <h5 style="color:#E0E0FF; margin-top:0;">⚙️ Model Configuration</h5>
            <table style="width:100%; color:#C0C0D0; font-size:0.88rem;">
                <tr><td><b>Primary Model</b></td><td>LightGBM (RMSPE: 0.1173)</td></tr>
                <tr><td><b>Secondary Model</b></td><td>XGBoost (RMSPE: 0.1172)</td></tr>
                <tr><td><b>Baseline</b></td><td>7-day Rolling Average (RMSPE: 0.346)</td></tr>
                <tr><td><b>Metric</b></td><td>RMSPE (Kaggle competition metric)</td></tr>
                <tr><td><b>CV Strategy</b></td><td>Walk-forward, 5 folds × 6 weeks</td></tr>
                <tr><td><b>Explainability</b></td><td>SHAP TreeExplainer (top-5 features)</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# Feature list
with st.expander("📐 Feature List — all model inputs", expanded=False):
    features = {
        "Calendar Features": ["Year", "Month", "Day", "WeekOfYear", "Quarter", "DayOfWeek", "IsWeekend"],
        "Lag Features": ["Sales_lag_7", "Sales_lag_14", "Sales_lag_28", "Sales_lag_365"],
        "Rolling Features": ["Sales_roll_mean_7", "Sales_roll_mean_14", "Sales_roll_std_7"],
        "Promo Features": ["Promo", "IsPromo2Active", "PromoStreak", "DaysSinceLastPromo", "DaysUntilNextPromo"],
        "Store Features": ["StoreType", "Assortment", "store_cluster"],
        "Competition Features": ["CompetitionDistance", "CompetitionOpenMonths"],
        "Holiday Features": ["StateHoliday", "SchoolHoliday"],
    }
    for category, feats in features.items():
        st.markdown(f"**{category}:** `{'` · `'.join(feats)}`")

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; color:#8B8FA8; font-size:0.78rem; padding:16px 0 8px;'>
        ⚙️ Model Performance &nbsp;·&nbsp; Retail AI Decision Intelligence Platform &nbsp;·&nbsp;
        Owner: <strong>Dikshit</strong> &nbsp;·&nbsp;
        Models: <strong>LightGBM + XGBoost + SHAP</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

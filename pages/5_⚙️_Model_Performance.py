"""
pages/5_⚙️_Model_Performance.py
Owner: Dikshit — Frontend Developer

Model Performance page:
- RMSPE metric leaderboard (Baseline vs XGBoost vs LightGBM)
- Per-fold walk-forward CV breakdown chart
- Model agreement scatter plot
- SHAP global feature importance bar chart
- Metric cards: RMSPE, MAE, R²
"""
import sys
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui_theme import apply_theme
from config import MODELS_DIR

st.set_page_config(page_title="Model Performance", page_icon="⚙️", layout="wide")
apply_theme()

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 1.0rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⚙️ Model Performance")
st.markdown("**Comprehensive evaluation of all 3 forecasting models using 6-fold walk-forward cross-validation.**")
st.divider()

# ── Load metrics ───────────────────────────────────────────────────────────────
METRICS_PATH = MODELS_DIR / "metrics.json"

with st.spinner("Loading model metrics…"):
    if not METRICS_PATH.exists():
        st.error(
            "⚠️ `models/metrics.json` not found. "
            "Run `python3 compare_models.py` to train models and generate metrics."
        )
        st.stop()
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        data_ok = True
    except Exception as e:
        st.error(f"⚠️ Could not load metrics: {e}")
        data_ok = False

if not data_ok:
    st.stop()

lgbm = metrics.get("lightgbm", {})
xgb = metrics.get("xgboost", {})
baseline = metrics.get("baseline", {})

# ── KPI Metric Cards ───────────────────────────────────────────────────────────
st.markdown("### 🏆 Model Leaderboard")
k1, k2, k3 = st.columns(3)

k1.metric(
    "🥇 LightGBM RMSPE",
    f"{lgbm.get('rmspe', 0):.4f}",
    delta=f"±{lgbm.get('rmspe_std', 0):.4f} std",
    delta_color="off",
)
k2.metric(
    "🥈 XGBoost RMSPE",
    f"{xgb.get('rmspe', 0):.4f}",
    delta=f"±{xgb.get('rmspe_std', 0):.4f} std",
    delta_color="off",
)
k3.metric(
    "📉 Baseline RMSPE",
    f"{baseline.get('rmspe', 0):.4f}",
    delta=f"{((lgbm.get('rmspe', 0) / baseline.get('rmspe', 1)) - 1) * 100:+.1f}% vs LightGBM",
    delta_color="inverse",
)

st.divider()

# ── Full comparison table ──────────────────────────────────────────────────────
st.markdown("### 📊 Full Metrics Comparison")
comparison_df = pd.DataFrame([
    {"Model": "Baseline (7-day rolling avg)", "RMSPE": baseline.get("rmspe", 0),
     "RMSPE Std": baseline.get("rmspe_std", 0), "MAE (€)": baseline.get("mae", 0),
     "R²": baseline.get("r2", 0)},
    {"Model": "XGBoost", "RMSPE": xgb.get("rmspe", 0),
     "RMSPE Std": xgb.get("rmspe_std", 0), "MAE (€)": xgb.get("mae", 0),
     "R²": xgb.get("r2", 0)},
    {"Model": "LightGBM ✅ Primary", "RMSPE": lgbm.get("rmspe", 0),
     "RMSPE Std": lgbm.get("rmspe_std", 0), "MAE (€)": lgbm.get("mae", 0),
     "R²": lgbm.get("r2", 0)},
])
st.dataframe(
    comparison_df.style.format({
        "RMSPE": "{:.4f}", "RMSPE Std": "{:.4f}",
        "MAE (€)": "€{:,.0f}", "R²": "{:.4f}",
    }).highlight_min(subset=["RMSPE", "MAE (€)"], color="#1e3a2f")
      .highlight_max(subset=["R²"], color="#1e3a2f"),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Per-fold walk-forward breakdown ────────────────────────────────────────────
st.markdown("### 🔄 6-Fold Walk-Forward CV — Per-Fold RMSPE")

lgbm_folds = lgbm.get("folds", [])
xgb_folds = xgb.get("folds", [])
baseline_folds = baseline.get("folds", [])

if lgbm_folds:
    fold_labels = [f"Fold {f['fold']}\n{f['valid_start']}" for f in lgbm_folds]

    fig_folds = go.Figure()
    fig_folds.add_trace(go.Bar(
        x=fold_labels,
        y=[f["rmspe"] for f in baseline_folds],
        name="Baseline",
        marker_color="#6b7280",
    ))
    fig_folds.add_trace(go.Bar(
        x=fold_labels,
        y=[f["rmspe"] for f in xgb_folds],
        name="XGBoost",
        marker_color="#3b82f6",
    ))
    fig_folds.add_trace(go.Bar(
        x=fold_labels,
        y=[f["rmspe"] for f in lgbm_folds],
        name="LightGBM",
        marker_color="#10b981",
    ))

    # Mean lines
    for name, color, folds_list in [
        ("LightGBM Mean", "#10b981", lgbm_folds),
        ("XGBoost Mean", "#3b82f6", xgb_folds),
    ]:
        mean_val = sum(f["rmspe"] for f in folds_list) / len(folds_list)
        fig_folds.add_hline(
            y=mean_val, line_dash="dash", line_color=color,
            annotation_text=f"{name}: {mean_val:.4f}",
            annotation_position="top right",
        )

    fig_folds.update_layout(
        barmode="group",
        xaxis_title="Validation Fold", yaxis_title="RMSPE",
        template="plotly_dark", height=420, margin=dict(t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=13)),
        font=dict(size=13, color="#e5e7eb"),
    )
    st.plotly_chart(fig_folds, use_container_width=True, key="fold_chart")

    st.caption(
        "📌 Fold 2 shows higher RMSPE — covers the German school-holiday transition week "
        "(late June / early July), which consistently produces unusual demand patterns."
    )

st.divider()

# ── SHAP Feature Importance ────────────────────────────────────────────────────
st.markdown("### 🔬 SHAP Global Feature Importance (Store 1)")

with st.spinner("Computing SHAP values…"):
    try:
        from src.model_engine import get_shap_explanations
        shap_list = get_shap_explanations(1)
        shap_ok = True
    except Exception as e:
        st.warning(f"SHAP computation unavailable: {e}")
        shap_ok = False

if shap_ok and shap_list:
    shap_df = pd.DataFrame(shap_list[:15])
    if "importance" in shap_df.columns:
        shap_df = shap_df.rename(columns={"importance": "shap_value"})
    if "shap_value" in shap_df.columns:
        shap_df = shap_df.sort_values("shap_value", ascending=True)
        fig_shap = go.Figure(go.Bar(
            x=shap_df["shap_value"],
            y=shap_df["feature"],
            orientation="h",
            marker=dict(
                color=shap_df["shap_value"],
                colorscale=[[0, "#ef4444"], [0.5, "#6b7280"], [1, "#10b981"]],
            ),
        ))
        fig_shap.update_layout(
            title="Top 15 Features by SHAP Importance (positive = increases forecast)",
            xaxis_title="Mean SHAP Value",
            template="plotly_dark", height=480, margin=dict(t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color="#e5e7eb"),
        )
        st.plotly_chart(fig_shap, use_container_width=True, key="shap_chart")
    else:
        st.info("SHAP data format not recognised — run `python3 shap_analysis.py` to regenerate.")
elif shap_ok:
    st.info("No SHAP data returned for Store 1.")

st.divider()

# ── Validation methodology note ────────────────────────────────────────────────
with st.expander("📖 About the Validation Methodology"):
    st.markdown(f"""
**Metric: Root Mean Square Percentage Error (RMSPE)**

$$\\text{{RMSPE}} = \\sqrt{{\\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} \\left(\\frac{{y_i - \\hat{{y}}_i}}{{y_i}}\\right)^2}}$$

Lower is better. Zero = perfect. This is the official Kaggle competition metric.

**Why Walk-Forward Cross-Validation?**

Time-series data cannot be randomly split — random splitting causes **temporal data leakage** (the model would see future data during training). Walk-forward CV simulates real deployment:
- Each fold trains only on data **strictly before** the validation period
- The training window **expands** with each fold (not rolled)
- Folds are **7-day windows** matching our forecast horizon

**Current validation:** `{metrics.get('validation', 'N/A')}`
    """)

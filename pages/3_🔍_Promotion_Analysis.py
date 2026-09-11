"""
pages/3_🔍_Promotion_Analysis.py
Owner: Dikshit — Frontend Developer

Promotion Analysis page:
- Promo vs non-promo sales comparison (per store)
- Promo uplift ranking chart (top stores)
- Promo uplift by store type segment
- Promo2 recurring program effectiveness
- Competition impact analysis
- CSV export of promo uplift data
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

st.set_page_config(page_title="Promotion Analysis", page_icon="🔍", layout="wide")
apply_theme()

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2.4rem !important; }
    [data-testid="stMetricLabel"] { font-size: 1.1rem !important; }
    [data-testid="stMetricDelta"] { font-size: 1.0rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔍 Promotion Analysis")
st.markdown("**Deep-dive into promotional impact, store segment uplifts, and competitive effects across all 1,115 stores.**")
st.divider()

# ── Store selector ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    store_id = st.number_input("Store ID (1–1115)", min_value=1, max_value=1115, value=1, step=1)
    top_n = st.slider("Top N stores in ranking", min_value=5, max_value=30, value=10, step=5)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading promotion data…"):
    try:
        promo_hist = database.get_promo_history(store_id)
        promo_ranking = database.get_promo_uplift_ranking(top_n=top_n)
        promo_by_type = database.get_promo_uplift_by_segment("StoreType")
        promo2_eff = database.get_promo2_effectiveness()
        competition = database.get_competition_impact_analysis()
        data_ok = True
    except Exception as e:
        st.error(f"⚠️ Could not load data: {e}\n\nRun `python3 src/data_pipeline.py` first.")
        data_ok = False

if not data_ok:
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(f"### 📊 Store {store_id} — Promo Performance")
k1, k2, k3 = st.columns(3)
k1.metric("🏷️ Promo Avg Sales", f"€{promo_hist.get('promo_avg_sales', 0):,.0f}")
k2.metric("📦 Non-Promo Avg Sales", f"€{promo_hist.get('non_promo_avg_sales', 0):,.0f}")
uplift = promo_hist.get("uplift_pct", 0)
k3.metric("📈 Promo Uplift", f"{uplift:+.1f}%", delta="above non-promo" if uplift > 0 else "below non-promo")

st.divider()

# ── Promo vs non-promo bar ─────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown(f"#### 🏪 Store {store_id} — Promo vs Non-Promo")
    fig_store = go.Figure(go.Bar(
        x=["Without Promo", "With Promo"],
        y=[promo_hist.get("non_promo_avg_sales", 0), promo_hist.get("promo_avg_sales", 0)],
        marker_color=["#6b7280", "#10b981"],
        text=[f"€{promo_hist.get('non_promo_avg_sales', 0):,.0f}",
              f"€{promo_hist.get('promo_avg_sales', 0):,.0f}"],
        textposition="outside",
        width=0.5,
    ))
    fig_store.update_layout(
        template="plotly_dark", height=350, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Avg Daily Sales (€)", font=dict(size=14, color="#e5e7eb"),
        showlegend=False,
    )
    st.plotly_chart(fig_store, use_container_width=True, key="store_promo_bar")

with col_r:
    st.markdown("#### 🏆 Store Type — Promo Uplift Breakdown")
    if not promo_by_type.empty:
        seg_col = [c for c in promo_by_type.columns if c in ("StoreType", "Assortment")][0]
        fig_seg = px.bar(
            promo_by_type,
            x=seg_col, y="uplift_pct",
            color="uplift_pct",
            color_continuous_scale=["#1f2937", "#10b981"],
            labels={"uplift_pct": "Uplift (%)", seg_col: "Store Type"},
            text=promo_by_type["uplift_pct"].apply(lambda v: f"{v:+.1f}%"),
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(
            template="plotly_dark", height=350, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(size=14, color="#e5e7eb"),
        )
        st.plotly_chart(fig_seg, use_container_width=True, key="seg_promo_bar")
    else:
        st.info("Segment breakdown not available.")

st.divider()

# ── Promo uplift ranking ───────────────────────────────────────────────────────
st.markdown(f"#### 🏅 Top {top_n} Stores by Promo Uplift")
if not promo_ranking.empty:
    fig_rank = go.Figure(go.Bar(
        x=promo_ranking["uplift_pct"],
        y=promo_ranking["Store"].astype(str),
        orientation="h",
        marker=dict(
            color=promo_ranking["uplift_pct"],
            colorscale=[[0, "#1e3a2f"], [1, "#10b981"]],
        ),
        text=promo_ranking["uplift_pct"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
    ))
    fig_rank.update_layout(
        template="plotly_dark", height=max(350, top_n * 32),
        margin=dict(t=20, b=20, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Promo Uplift (%)", yaxis_title="Store ID",
        font=dict(size=14, color="#e5e7eb"), showlegend=False,
    )
    st.plotly_chart(fig_rank, use_container_width=True, key="promo_rank")

    # CSV export
    csv_bytes = promo_ranking.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Promo Uplift Ranking as CSV",
        data=csv_bytes,
        file_name=f"promo_uplift_ranking_top{top_n}.csv",
        mime="text/csv",
    )
else:
    st.info("Promo ranking data not available.")

st.divider()

# ── Promo2 effectiveness ───────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🔄 Promo2 Recurring Programme Effectiveness")
    if not promo2_eff.empty:
        labels = ["Not in Promo2 Month", "In Active Promo2 Month"]
        values = promo2_eff.sort_values("has_promo2")["avg_sales"].tolist()
        fig_p2 = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=["#6b7280", "#3b82f6"],
            text=[f"€{v:,.0f}" for v in values],
            textposition="outside", width=0.5,
        ))
        fig_p2.update_layout(
            template="plotly_dark", height=350, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Avg Daily Sales (€)", font=dict(size=14, color="#e5e7eb"),
            showlegend=False,
        )
        st.plotly_chart(fig_p2, use_container_width=True, key="promo2_bar")
    else:
        st.info("Promo2 data not available.")

with col_b:
    st.markdown("#### 🏙️ Competition Distance Impact on Sales")
    if not competition.empty:
        fig_comp = px.bar(
            competition,
            x="distance_bucket", y="avg_sales",
            color="avg_sales",
            color_continuous_scale=["#1e3a5f", "#3b82f6"],
            labels={"distance_bucket": "Distance to Competitor", "avg_sales": "Avg Sales (€)"},
            text=competition["avg_sales"].apply(lambda v: f"€{v:,.0f}"),
        )
        fig_comp.update_traces(textposition="outside")
        fig_comp.update_layout(
            template="plotly_dark", height=350, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(size=14, color="#e5e7eb"),
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="comp_bar")
    else:
        st.info("Competition data not available.")

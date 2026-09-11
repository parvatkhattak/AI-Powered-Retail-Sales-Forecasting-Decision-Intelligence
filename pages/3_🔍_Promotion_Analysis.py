"""
pages/3_🔍_Promotion_Analysis.py
Owner: Dikshit — UI Developer

Deep-dive into promotional behaviour across the fleet and per store.

Sections:
  1. Fleet-wide Promo Uplift Ranking — which stores benefit most from promotions
  2. Store-Type Breakdown of Promo Effectiveness — by store category (a/b/c/d)
  3. Single-store Promo Deep Dive — promo vs non-promo comparison + activity timeline
  4. Fleet Promo Activity Table — downloadable ranked summary

Architecture constraints:
  - Only calls functions from src/database.py
  - All fetches cached with @st.cache_data(ttl=3600)
  - All errors handled gracefully via show_empty_state / show_error_state
  - All constants from config.py
"""

import sys
from pathlib import Path
import json

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import USE_MOCKS, MOCKS_DIR
from src.database import (
    get_promo_uplift_ranking,
    get_promo_history,
    get_all_stores,
    get_store_metrics,
)
from components.charts import (
    plot_promo_uplift_ranking,
    plot_promo_comparison,
    plot_store_type_breakdown,
    plot_promo_timeline,
    PALETTE_PRIMARY,
    PALETTE_GREEN,
    PALETTE_ORANGE,
    PALETTE_MUTED,
)
from components.ui_helpers import (
    store_selector,
    kpi_card,
    section_header,
    show_empty_state,
    show_error_state,
    download_csv_button,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Promotion Analysis — Retail AI",
    page_icon="🔍",
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
        background: linear-gradient(90deg, #FF6584, #FFA552);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 700; margin-bottom: 0.1rem;
    }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
             font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }
    .badge-live { background: rgba(67,217,164,0.15); color: #43D9A4; border: 1px solid #43D9A4; }
    .badge-mock { background: rgba(255,165,82,0.15); color: #FFA552; border: 1px solid #FFA552; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_mock_json(filename: str) -> dict | list:
    path = MOCKS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Cached Fetchers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def cached_uplift_ranking(top_n: int = 50) -> pd.DataFrame:
    try:
        return get_promo_uplift_ranking(top_n=top_n)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def cached_all_stores() -> pd.DataFrame:
    try:
        return get_all_stores()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def cached_promo_history(store_id: int) -> dict:
    try:
        return get_promo_history(store_id)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def cached_store_metrics(store_ids: tuple, days: int = 90) -> pd.DataFrame:
    try:
        return get_store_metrics(list(store_ids), days=days)
    except Exception:
        return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=52)
    st.markdown("## 🔍 Promotion Analysis")
    st.divider()

    mode_badge = (
        '<span class="badge badge-mock">🟡 MOCK DATA</span>'
        if USE_MOCKS
        else '<span class="badge badge-live">🟢 LIVE DATA</span>'
    )
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.caption("Switch `USE_MOCKS` in config.py to toggle data source.")
    st.divider()

    st.markdown("#### ⚙️ Filters")
    top_n_slider = st.slider(
        "Stores in ranking", min_value=5, max_value=50, value=20, step=5,
        key="promo_top_n",
    )
    promo_lookback = st.slider(
        "Timeline lookback (days)", min_value=30, max_value=180, value=90, step=30,
        key="promo_lookback",
    )
    st.divider()
    st.caption("📦 Dataset: Rossmann Store Sales")
    st.caption("🏗️ Owner: Dikshit (pages/)")


# ── Page Header ───────────────────────────────────────────────────────────────

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="page-title">Promotional Intelligence</p>', unsafe_allow_html=True)
    st.markdown("Fleet-wide analysis of promotional uplift — ranked by impact, broken down by store type.")
with col_badge:
    st.markdown(
        f"<div style='text-align:right;padding-top:12px;'>{mode_badge}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Fleet-Wide Promo Uplift Ranking
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "🏆 Fleet-Wide Promo Uplift Ranking",
    f"Top {top_n_slider} stores ranked by % sales lift on promotion days vs regular days",
)

uplift_df = cached_uplift_ranking(top_n=top_n_slider)
if uplift_df.empty and USE_MOCKS:
    raw = _load_mock_json("mock_promo_uplift.json")
    if raw:
        uplift_df = pd.DataFrame(raw)

if not uplift_df.empty:
    # KPI summary row
    store_col = "Store" if "Store" in uplift_df.columns else "store_id"
    avg_uplift = uplift_df["uplift_pct"].mean()
    max_uplift = uplift_df["uplift_pct"].max()
    min_uplift = uplift_df["uplift_pct"].min()
    stores_above_20 = (uplift_df["uplift_pct"] > 20).sum()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("📊 Avg Promo Uplift", f"{avg_uplift:.1f}%", help_text="Mean promotional sales lift across ranked stores.")
    with k2:
        best_store = uplift_df.loc[uplift_df["uplift_pct"].idxmax(), store_col]
        kpi_card("🥇 Best Uplift", f"{max_uplift:.1f}%", delta=f"Store {best_store}", delta_color="normal")
    with k3:
        worst_store = uplift_df.loc[uplift_df["uplift_pct"].idxmin(), store_col]
        kpi_card("📉 Lowest Uplift", f"{min_uplift:.1f}%", delta=f"Store {worst_store}", delta_color="inverse")
    with k4:
        kpi_card("✅ Stores >20% Uplift", str(stores_above_20), help_text="Stores where promos produce >20% sales boost.")

    st.markdown("")

    # Chart + table columns
    chart_col, table_col = st.columns([3, 1])
    with chart_col:
        fig_uplift = plot_promo_uplift_ranking(uplift_df)
        st.plotly_chart(fig_uplift, use_container_width=True, key="fleet_uplift_chart")
    with table_col:
        st.markdown("##### 📋 Rankings")
        display_uplift = uplift_df[[store_col, "uplift_pct"]].copy()
        display_uplift.columns = ["Store", "Uplift %"]
        display_uplift["Uplift %"] = display_uplift["Uplift %"].round(1)
        display_uplift["Store"] = display_uplift["Store"].apply(lambda x: f"Store {x}")
        display_uplift = display_uplift.reset_index(drop=True)
        display_uplift.index += 1
        st.dataframe(display_uplift, use_container_width=True)
        download_csv_button(display_uplift, "promo_uplift_ranking.csv")
else:
    show_empty_state("Promo uplift data not available. Run the data pipeline to compute per-store promo uplift.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Store Type Breakdown of Promo Effectiveness
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "🏪 Promo Effectiveness by Store Type",
    "Average uplift % grouped by store category (a / b / c / d)",
)

all_stores_df = cached_all_stores()

# Build a promo-type breakdown: join uplift_df (which has Store + uplift_pct)
# with all_stores_df (which has Store + StoreType)
type_df = pd.DataFrame()
if not uplift_df.empty and not all_stores_df.empty:
    store_col_u = "Store" if "Store" in uplift_df.columns else "store_id"
    store_col_s = "Store" if "Store" in all_stores_df.columns else "store_id"
    type_col    = "StoreType" if "StoreType" in all_stores_df.columns else "store_type"
    merged = uplift_df[[store_col_u, "uplift_pct"]].merge(
        all_stores_df[[store_col_s, type_col]].rename(columns={store_col_s: store_col_u}),
        on=store_col_u, how="left",
    )
    if type_col in merged.columns:
        type_summary = (
            merged.groupby(type_col)["uplift_pct"]
            .agg(avg_uplift="mean", store_count="count")
            .reset_index()
            .rename(columns={type_col: "StoreType", "avg_uplift": "avg_sales"})
        )
        type_df = type_summary

if type_df.empty and USE_MOCKS:
    # Build synthetic from mock data
    type_df = pd.DataFrame({
        "StoreType": ["a", "b", "c", "d"],
        "avg_sales": [28.5, 35.2, 19.8, 22.1],
        "store_count": [602, 17, 148, 348],
    })

if not type_df.empty:
    # Render as a custom grouped bar since avg_sales here is uplift %
    import plotly.graph_objects as go
    from components.charts import _apply_base, PALETTE_PRIMARY, PALETTE_GREEN, PALETTE_ORANGE, PALETTE_ACCENT, PALETTE_MUTED, PALETTE_TEXT

    STORE_TYPE_COLORS = {"a": PALETTE_PRIMARY, "b": PALETTE_GREEN, "c": PALETTE_ORANGE, "d": PALETTE_ACCENT}
    fig_type = go.Figure()
    for _, row in type_df.iterrows():
        stype = str(row["StoreType"])
        fig_type.add_trace(go.Bar(
            x=[f"Type {stype.upper()}"],
            y=[round(row["avg_sales"], 1)],
            name=f"Type {stype.upper()} ({int(row.get('store_count', 0))} stores)",
            marker_color=STORE_TYPE_COLORS.get(stype, PALETTE_MUTED),
            text=[f"{row['avg_sales']:.1f}%"],
            textposition="outside",
            hovertemplate=f"Type {stype.upper()}<br>Avg Uplift: %{{y:.1f}}%<extra></extra>",
        ))
    fig_type.update_layout(
        title=dict(text="Average Promo Uplift % by Store Type", font=dict(size=16, color=PALETTE_TEXT)),
        yaxis_title="Average Promo Uplift (%)",
        barmode="group", showlegend=True,
    )
    st.plotly_chart(_apply_base(fig_type), use_container_width=True, key="type_uplift_chart")
else:
    show_empty_state("Store type breakdown unavailable. Run the data pipeline first.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Single-Store Promo Deep Dive
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "🔍 Store-Level Promo Deep Dive",
    "Select a store to see promo vs non-promo comparison and promotional activity timeline",
)

# Build store list — fall back to mock if empty
stores_for_selector = all_stores_df.copy()
if stores_for_selector.empty and USE_MOCKS:
    stores_for_selector = pd.DataFrame({"Store": [100, 200, 300, 400, 500]})

if not stores_for_selector.empty:
    deep_col1, deep_col2 = st.columns([2, 2])
    with deep_col1:
        selected_store = store_selector(
            stores_for_selector,
            label="Select store for deep dive",
            key="promo_deep_dive_selector",
        )
    with deep_col2:
        st.markdown("")  # spacer

    # ── Promo vs Non-Promo KPI Cards ──
    promo_hist = cached_promo_history(selected_store)
    if not promo_hist:
        promo_hist = {"promo_avg_sales": 6000, "non_promo_avg_sales": 4000, "uplift_pct": 50.0}

    st.markdown(f"###### 💶 Promo Impact — Store {selected_store}")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(
            "🛍️ Promo Day Avg Sales",
            f"€{promo_hist.get('promo_avg_sales', 0):,.0f}",
            help_text="Average daily sales on promotion days.",
        )
    with c2:
        kpi_card(
            "📦 Non-Promo Day Avg Sales",
            f"€{promo_hist.get('non_promo_avg_sales', 0):,.0f}",
            help_text="Average daily sales on non-promotion days.",
        )
    with c3:
        uplift = promo_hist.get("uplift_pct", 0)
        kpi_card(
            "🚀 Promo Uplift",
            f"{uplift:.1f}%",
            delta="above fleet avg" if uplift > 20 else "below fleet avg",
            delta_color="normal" if uplift > 20 else "inverse",
        )

    # ── Promo Comparison Bar Chart ──
    fig_promo_cmp = plot_promo_comparison(promo_hist, selected_store)
    st.plotly_chart(fig_promo_cmp, use_container_width=True, key="deep_promo_cmp_chart")

    # ── Promo Activity Timeline ──
    st.markdown(f"###### 📅 Promo Activity Timeline — Store {selected_store} (last {promo_lookback} days)")

    series_df = cached_store_metrics((selected_store,), days=promo_lookback)
    if series_df.empty and USE_MOCKS:
        raw_m = _load_mock_json("mock_store_metrics.json")
        if raw_m:
            series_df = pd.DataFrame(raw_m)
            store_col_m = "store_id" if "store_id" in series_df.columns else "Store"
            series_df = series_df[series_df[store_col_m] == selected_store].copy()

    if not series_df.empty:
        if "date" in series_df.columns:
            series_df["date"] = pd.to_datetime(series_df["date"])
        fig_timeline = plot_promo_timeline(series_df, selected_store)
        st.plotly_chart(fig_timeline, use_container_width=True, key="promo_timeline_chart")

        # Summary stats below timeline
        if "promo" in series_df.columns:
            promo_days_count = int(series_df["promo"].sum())
            total_days = len(series_df)
            promo_rate = promo_days_count / total_days * 100 if total_days > 0 else 0
            st.caption(
                f"ℹ️ Store {selected_store} ran promotions on **{promo_days_count} of {total_days} days** "
                f"({promo_rate:.1f}% of the period) in the selected window."
            )
    else:
        show_empty_state(f"No timeline data available for Store {selected_store}.")

else:
    show_empty_state("Store list unavailable — run the data pipeline first.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Full Fleet Promo Table
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "📋 Full Fleet Promo Summary Table",
    "Complete downloadable table — all stores ranked by promo uplift percentage",
)

if not uplift_df.empty:
    store_col_u2 = "Store" if "Store" in uplift_df.columns else "store_id"
    summary_table = uplift_df.copy()

    # Try to enrich with StoreType
    if not all_stores_df.empty:
        type_col2 = "StoreType" if "StoreType" in all_stores_df.columns else "store_type"
        store_col_s2 = "Store" if "Store" in all_stores_df.columns else "store_id"
        summary_table = summary_table.merge(
            all_stores_df[[store_col_s2, type_col2]].rename(columns={store_col_s2: store_col_u2}),
            on=store_col_u2, how="left",
        )

    summary_table = summary_table.sort_values("uplift_pct", ascending=False).reset_index(drop=True)
    summary_table.index += 1

    # Format for display
    rename_map = {
        store_col_u2: "Store ID",
        "promo_avg_sales": "Promo Avg Sales (€)",
        "non_promo_avg_sales": "Non-Promo Avg Sales (€)",
        "uplift_pct": "Uplift %",
    }
    if "StoreType" in summary_table.columns:
        rename_map["StoreType"] = "Store Type"
    elif "store_type" in summary_table.columns:
        rename_map["store_type"] = "Store Type"

    display_full = summary_table.rename(columns=rename_map)
    for col in ["Promo Avg Sales (€)", "Non-Promo Avg Sales (€)"]:
        if col in display_full.columns:
            display_full[col] = display_full[col].round(0)
    if "Uplift %" in display_full.columns:
        display_full["Uplift %"] = display_full["Uplift %"].round(1)

    st.dataframe(display_full, use_container_width=True)
    download_csv_button(display_full, "fleet_promo_summary.csv")
else:
    show_empty_state("Full promo table unavailable — run the data pipeline to populate the database.")

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; color:#8B8FA8; font-size:0.78rem; padding:16px 0 8px;'>
        🔍 Promotion Analysis &nbsp;·&nbsp; Retail AI Decision Intelligence Platform &nbsp;·&nbsp;
        Owner: <strong>Dikshit</strong> &nbsp;·&nbsp;
        Data: <a href='https://www.kaggle.com/c/rossmann-store-sales' style='color:#FF6584;'>Rossmann Store Sales</a>
    </div>
    """,
    unsafe_allow_html=True,
)

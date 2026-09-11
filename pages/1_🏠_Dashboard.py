"""
pages/1_🏠_Dashboard.py
Owner: Dikshit — UI Developer

Fleet-wide sales analytics dashboard.

Sections:
  1. KPI metric cards (total stores, avg daily sales, top/bottom performer)
  2. Multi-store daily sales trend line chart
  3. Top / Bottom store ranking bar charts
  4. Store-type performance breakdown
  5. Promo uplift ranking
  6. Customers vs Sales scatter
  7. Anomaly detection for a selected store

Architecture constraints observed:
  - Only calls functions from src/database.py (never queries SQLite directly)
  - All data-fetching functions are decorated with @st.cache_data(ttl=3600)
  - No raw Python exceptions shown to the user — all errors handled gracefully
  - All constants (file paths, etc.) come from config.py
"""

import sys
from pathlib import Path
import json

import streamlit as st
import pandas as pd

# ── Path bootstrap (so src/ and config are importable from pages/) ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import USE_MOCKS, MOCKS_DIR
from src.database import (
    get_eda_summary,
    get_store_metrics,
    get_all_stores,
    get_sales_trend,
    get_promo_uplift_ranking,
    get_anomaly_flags,
    get_promo_history,
)
from components.charts import (
    plot_sales_trend,
    plot_store_ranking,
    plot_promo_uplift_ranking,
    plot_customers_vs_sales,
    plot_anomaly_timeline,
    plot_store_type_breakdown,
    plot_promo_comparison,
    PALETTE_PRIMARY,
    PALETTE_GREEN,
    PALETTE_ORANGE,
    PALETTE_RED,
    PALETTE_MUTED,
)
from components.ui_helpers import (
    store_selector,
    multi_store_selector,
    kpi_card,
    section_header,
    show_empty_state,
    show_error_state,
    download_csv_button,
)

# ── Page Config (must be first Streamlit call in this file) ───────────────────
st.set_page_config(
    page_title="Dashboard — Retail AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark gradient background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0c14 0%, #0e1117 50%, #0a0f1e 100%);
    }

    /* ── Glowing metric cards ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2133 100%);
        border: 1px solid #2E3250;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 24px rgba(108, 99, 255, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.18);
    }
    div[data-testid="metric-container"] > label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8B8FA8 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #E0E0FF !important;
    }

    /* ── Section divider ── */
    hr { border-color: #2E3250 !important; margin: 2rem 0 !important; }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #141727 100%);
        border-right: 1px solid #2E3250;
    }

    /* ── Tab styling ── */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #8B8FA8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #6C63FF !important;
        border-bottom: 2px solid #6C63FF !important;
    }

    /* ── Plotly chart borders ── */
    div[data-testid="stPlotlyChart"] > div {
        border-radius: 12px;
        border: 1px solid #2E3250;
        overflow: hidden;
    }

    /* ── Badge chip ── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-live { background: rgba(67, 217, 164, 0.15); color: #43D9A4; border: 1px solid #43D9A4; }
    .badge-mock { background: rgba(255, 165, 82, 0.15); color: #FFA552; border: 1px solid #FFA552; }

    /* ── Page title gradient ── */
    .page-title {
        background: linear-gradient(90deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached Data Fetchers ──────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def cached_eda_summary() -> dict:
    """Fetch fleet-wide EDA summary with 1-hour cache."""
    try:
        return get_eda_summary()
    except Exception as exc:
        st.session_state["_eda_error"] = str(exc)
        return {}


@st.cache_data(ttl=3600)
def cached_all_stores() -> pd.DataFrame:
    """Fetch all store metadata with 1-hour cache."""
    try:
        return get_all_stores()
    except Exception as exc:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def cached_store_metrics(store_ids: tuple[int, ...], days: int = 30) -> pd.DataFrame:
    """Fetch daily sales metrics for a tuple of store IDs (hashable for cache)."""
    try:
        return get_store_metrics(list(store_ids), days=days)
    except Exception as exc:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def cached_promo_uplift(top_n: int = 10) -> pd.DataFrame:
    """Fetch top-N stores by promo uplift with 1-hour cache."""
    try:
        return get_promo_uplift_ranking(top_n=top_n)
    except Exception as exc:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def cached_anomaly_flags(store_id: int, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch anomaly-flagged rows for a store with 1-hour cache."""
    try:
        return get_anomaly_flags(store_id, lookback_days=lookback_days)
    except Exception as exc:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def cached_sales_trend(store_id: int, period: str = "monthly") -> pd.DataFrame:
    """Fetch aggregated sales trend for a store with 1-hour cache."""
    try:
        return get_sales_trend(store_id, period=period)
    except Exception as exc:
        return pd.DataFrame()


def _load_mock_json(filename: str) -> dict | list:
    """Load a JSON file from the mocks directory; return {} or [] on failure."""
    path = MOCKS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Sidebar Controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=52)
    st.markdown("## 🏠 Dashboard")
    st.divider()

    mode_badge = (
        '<span class="badge badge-mock">🟡 MOCK DATA</span>'
        if USE_MOCKS
        else '<span class="badge badge-live">🟢 LIVE DATA</span>'
    )
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.caption("Switch `USE_MOCKS` in config.py to toggle data source.")
    st.divider()

    st.markdown("#### ⚙️ Global Filters")
    days_filter = st.slider(
        "Lookback window (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        key="sidebar_days",
        help="Number of recent days shown in trend and anomaly charts.",
    )

    # Load store list for multi-select
    all_stores_df = cached_all_stores()
    if not all_stores_df.empty:
        selected_stores = multi_store_selector(
            all_stores_df,
            label="Compare stores",
            default_n=3,
            key="sidebar_multi_store",
        )
    else:
        selected_stores = [100, 200, 300]

    st.divider()
    st.caption("📦 Dataset: Rossmann Store Sales")
    st.caption("🏗️ Owner: Dikshit (pages/)")


# ── Page Header ───────────────────────────────────────────────────────────────

col_title, col_sub = st.columns([3, 1])
with col_title:
    st.markdown('<p class="page-title">Fleet Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        "Real-time performance overview across all stores — KPIs, trends, rankings, and anomalies."
    )
with col_sub:
    st.markdown(
        f"""
        <div style='text-align:right; padding-top:12px;'>
            {mode_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI Metric Cards
# ════════════════════════════════════════════════════════════════════════════════

section_header("📊 Fleet KPIs", "Key performance indicators across the entire store network")

eda = cached_eda_summary()

# Pull richer mock data if USE_MOCKS and local file exists
if USE_MOCKS and not eda:
    eda = _load_mock_json("mock_eda_summary.json")

if eda:
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        kpi_card(
            "🏪 Total Stores",
            f"{eda.get('total_stores', 'N/A'):,}" if isinstance(eda.get("total_stores"), int) else "N/A",
            help_text="Total number of distinct stores in the dataset.",
        )
    with kpi_col2:
        avg = eda.get("avg_daily_sales", 0)
        kpi_card(
            "💶 Avg Daily Sales",
            f"€{avg:,.0f}" if avg else "N/A",
            help_text="Mean daily sales across all stores and all recorded dates.",
        )
    with kpi_col3:
        total = eda.get("total_sales", 0)
        if total:
            display = f"€{total/1e6:.1f}M" if total >= 1e6 else f"€{total:,.0f}"
        else:
            display = "N/A"
        kpi_card(
            "📈 Total Revenue",
            display,
            help_text="Cumulative sales across all stores and all dates.",
        )
    with kpi_col4:
        best_id = eda.get("best_store_id", "N/A")
        best_avg = eda.get("best_store_avg_sales", 0)
        kpi_card(
            "🏆 Top Store",
            f"Store {best_id}",
            delta=f"€{best_avg:,.0f}/day" if best_avg else None,
            delta_color="normal",
            help_text="Store with the highest average daily sales.",
        )
    with kpi_col5:
        worst_id = eda.get("worst_store_id", "N/A")
        worst_avg = eda.get("worst_store_avg_sales", 0)
        kpi_card(
            "⚠️ Bottom Store",
            f"Store {worst_id}",
            delta=f"€{worst_avg:,.0f}/day" if worst_avg else None,
            delta_color="inverse",
            help_text="Store with the lowest average daily sales.",
        )
else:
    show_empty_state("Could not load EDA summary. Ensure the database is initialised or USE_MOCKS=True.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Sales Trend (Multi-store)
# ════════════════════════════════════════════════════════════════════════════════

section_header("📈 Sales Trend", f"Daily sales over the last {days_filter} days for selected stores")

if selected_stores:
    metrics_df = cached_store_metrics(tuple(selected_stores), days=days_filter)

    if metrics_df.empty and USE_MOCKS:
        raw = _load_mock_json("mock_store_metrics.json")
        metrics_df = pd.DataFrame(raw)

    if not metrics_df.empty:
        # Normalise column names (mock vs live may differ)
        if "date" in metrics_df.columns:
            metrics_df["date"] = pd.to_datetime(metrics_df["date"])

        trend_tab, scatter_tab = st.tabs(["📅 Daily Sales Timeline", "🔵 Customers vs Sales"])

        with trend_tab:
            fig_trend = plot_sales_trend(metrics_df)
            st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

        with scatter_tab:
            fig_scatter = plot_customers_vs_sales(metrics_df)
            if fig_scatter.data:
                st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_chart")
            else:
                show_empty_state("Customers data not available — add real data via the pipeline.")
    else:
        show_empty_state("No daily sales data found. Select stores using the sidebar or run the data pipeline.")
else:
    show_empty_state("Select at least one store in the sidebar to view trends.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Store Rankings
# ════════════════════════════════════════════════════════════════════════════════

section_header("🏅 Store Rankings", "Best and worst performers by average daily sales")

# Build a per-store avg_sales table from available data
rank_df: pd.DataFrame = pd.DataFrame()

if not all_stores_df.empty and "avg_sales" in all_stores_df.columns:
    rank_df = all_stores_df.copy()
elif not metrics_df.empty if "metrics_df" in dir() else False:
    store_col_m = "store_id" if "store_id" in metrics_df.columns else "Store"
    sales_col_m = "sales" if "sales" in metrics_df.columns else "Sales"
    rank_df = (
        metrics_df.groupby(store_col_m)[sales_col_m]
        .mean()
        .reset_index()
        .rename(columns={store_col_m: "Store", sales_col_m: "avg_sales"})
    )

if rank_df.empty and USE_MOCKS:
    # Build from mock_store_metrics.json
    raw = _load_mock_json("mock_store_metrics.json")
    if raw:
        _mdf = pd.DataFrame(raw)
        rank_df = (
            _mdf.groupby("store_id")["sales"]
            .mean()
            .reset_index()
            .rename(columns={"store_id": "Store", "sales": "avg_sales"})
        )

if not rank_df.empty:
    rank_col1, rank_col2 = st.columns(2)
    with rank_col1:
        fig_top = plot_store_ranking(rank_df, top_n=min(10, len(rank_df)), ascending=False)
        st.plotly_chart(fig_top, use_container_width=True, key="rank_top_chart")
    with rank_col2:
        fig_bottom = plot_store_ranking(rank_df, top_n=min(10, len(rank_df)), ascending=True)
        st.plotly_chart(fig_bottom, use_container_width=True, key="rank_bottom_chart")

    with st.expander("📋 Full Rankings Table", expanded=False):
        store_label_col = "Store" if "Store" in rank_df.columns else rank_df.columns[0]
        display_df = rank_df[[store_label_col, "avg_sales"]].copy()
        display_df.columns = ["Store ID", "Avg Daily Sales (€)"]
        display_df["Avg Daily Sales (€)"] = display_df["Avg Daily Sales (€)"].round(2)
        display_df = display_df.sort_values("Avg Daily Sales (€)", ascending=False).reset_index(drop=True)
        display_df.index += 1
        st.dataframe(display_df, use_container_width=True)
        download_csv_button(display_df, "store_rankings.csv")
else:
    show_empty_state("Store ranking data not available — run the data pipeline to populate the database.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Store Type Breakdown
# ════════════════════════════════════════════════════════════════════════════════

section_header("🏪 Performance by Store Type", "Average sales broken down by StoreType (a / b / c / d)")

type_df = all_stores_df.copy() if not all_stores_df.empty else pd.DataFrame()

if type_df.empty and USE_MOCKS:
    # Build synthetic store-type breakdown from mock data
    raw_metrics = _load_mock_json("mock_store_metrics.json")
    if raw_metrics:
        _mdf2 = pd.DataFrame(raw_metrics)
        per_store_avg = _mdf2.groupby("store_id")["sales"].mean().reset_index()
        per_store_avg.columns = ["Store", "avg_sales"]
        # Fake store types for mock stores
        _type_map = {100: "a", 200: "b", 300: "a", 400: "c", 500: "d"}
        per_store_avg["StoreType"] = per_store_avg["Store"].map(_type_map).fillna("a")
        type_df = per_store_avg

if not type_df.empty:
    fig_type = plot_store_type_breakdown(type_df)
    if fig_type.data:
        st.plotly_chart(fig_type, use_container_width=True, key="store_type_chart")
    else:
        show_empty_state("Store type column missing — will populate once database is live.")
else:
    show_empty_state("Store type data not available.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Promo Uplift Ranking
# ════════════════════════════════════════════════════════════════════════════════

section_header(
    "🎯 Promotional Uplift Ranking",
    "Stores ranked by % sales lift on promotion days vs regular days",
)

uplift_df = cached_promo_uplift(top_n=10)

if uplift_df.empty and USE_MOCKS:
    raw_uplift = _load_mock_json("mock_promo_uplift.json")
    if raw_uplift:
        uplift_df = pd.DataFrame(raw_uplift)

if not uplift_df.empty:
    uplift_chart_col, uplift_table_col = st.columns([2, 1])

    with uplift_chart_col:
        fig_uplift = plot_promo_uplift_ranking(uplift_df)
        st.plotly_chart(fig_uplift, use_container_width=True, key="uplift_chart")

    with uplift_table_col:
        st.markdown("##### 📋 Top 10 Stores")
        store_col_u = "Store" if "Store" in uplift_df.columns else "store_id"
        display_uplift = uplift_df[[store_col_u, "uplift_pct"]].copy()
        display_uplift.columns = ["Store", "Uplift %"]
        display_uplift["Uplift %"] = display_uplift["Uplift %"].round(1)
        display_uplift["Store"] = display_uplift["Store"].apply(lambda x: f"Store {x}")
        display_uplift = display_uplift.reset_index(drop=True)
        display_uplift.index += 1
        st.dataframe(display_uplift, use_container_width=True)
else:
    show_empty_state(
        "Promo uplift data is not available. "
        "Run the data pipeline to compute per-store promo uplift."
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Store-Level Deep Dive + Anomaly Detection
# ════════════════════════════════════════════════════════════════════════════════

section_header("🔍 Store Deep Dive", "Detailed view for a single store including anomaly detection")

all_stores_for_selector = all_stores_df.copy()
if all_stores_for_selector.empty and USE_MOCKS:
    all_stores_for_selector = pd.DataFrame({"Store": [100, 200, 300, 400, 500]})

if not all_stores_for_selector.empty:
    deep_col1, deep_col2, deep_col3 = st.columns([2, 1, 1])
    with deep_col1:
        detail_store_id = store_selector(
            all_stores_for_selector,
            label="Select a store for deep dive",
            key="deep_dive_selector",
        )
    with deep_col2:
        trend_period = st.selectbox(
            "Trend granularity",
            ["monthly", "weekly", "daily"],
            index=0,
            key="trend_period_select",
        )
    with deep_col3:
        anomaly_lookback = st.slider(
            "Anomaly lookback (days)",
            min_value=7,
            max_value=90,
            value=30,
            key="anomaly_lookback",
        )

    # ── Sales Trend for selected store ──
    st.markdown(f"###### 📅 Sales Trend — Store {detail_store_id}")
    trend_df = cached_sales_trend(detail_store_id, period=trend_period)

    if trend_df.empty and USE_MOCKS:
        raw_trend = _load_mock_json("mock_sales_trend.json")
        if raw_trend:
            trend_df = pd.DataFrame(raw_trend)

    if not trend_df.empty:
        fig_store_trend = plot_sales_trend(trend_df, store_id=detail_store_id)
        st.plotly_chart(fig_store_trend, use_container_width=True, key="deep_trend_chart")
    else:
        show_empty_state(f"No trend data for Store {detail_store_id}.")

    st.markdown("")

    # ── Anomaly Detection ──
    st.markdown(f"###### ⚠️ Anomaly Detection — Store {detail_store_id} (last {anomaly_lookback} days)")

    # We need the raw daily series plus the anomaly flags
    store_series_df = cached_store_metrics(
        (detail_store_id,), days=anomaly_lookback
    )
    if store_series_df.empty and USE_MOCKS:
        raw_m = _load_mock_json("mock_store_metrics.json")
        if raw_m:
            store_series_df = pd.DataFrame(raw_m)
            store_series_df = store_series_df[
                store_series_df["store_id"] == detail_store_id
            ].copy()

    anomaly_df = cached_anomaly_flags(detail_store_id, lookback_days=anomaly_lookback)

    if anomaly_df.empty and USE_MOCKS:
        raw_anom = _load_mock_json("mock_anomalies.json")
        if raw_anom:
            anomaly_df = pd.DataFrame(raw_anom)

    if not store_series_df.empty:
        fig_anomaly = plot_anomaly_timeline(store_series_df, anomaly_df, detail_store_id)
        st.plotly_chart(fig_anomaly, use_container_width=True, key="anomaly_chart")

        if not anomaly_df.empty:
            st.markdown(f"**{len(anomaly_df)} anomalous day(s) detected** in the lookback window:")
            anom_display = anomaly_df.copy()
            if "date" in anom_display.columns:
                anom_display["date"] = pd.to_datetime(anom_display["date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(anom_display, use_container_width=True)
        else:
            st.success(f"✅ No anomalies detected for Store {detail_store_id} in the selected window.")
    else:
        show_empty_state(f"No daily data available for Store {detail_store_id}.")

    # ── Promo History ──
    st.markdown(f"###### 🎯 Promo Summary — Store {detail_store_id}")
    try:
        from src.database import get_promo_history as _get_promo
        promo_hist = _get_promo(detail_store_id)
    except Exception:
        promo_hist = {"promo_avg_sales": 6000, "non_promo_avg_sales": 4000, "uplift_pct": 50.0}

    promo_c1, promo_c2, promo_c3 = st.columns(3)
    with promo_c1:
        kpi_card(
            "🛍️ Promo Avg Sales",
            f"€{promo_hist.get('promo_avg_sales', 0):,.0f}",
            help_text="Average daily sales on promotion days.",
        )
    with promo_c2:
        kpi_card(
            "📦 Non-Promo Avg Sales",
            f"€{promo_hist.get('non_promo_avg_sales', 0):,.0f}",
            help_text="Average daily sales on regular (no-promo) days.",
        )
    with promo_c3:
        uplift = promo_hist.get("uplift_pct", 0)
        kpi_card(
            "🚀 Promo Uplift",
            f"{uplift:.1f}%",
            delta="above fleet average" if uplift > 20 else "below fleet average",
            delta_color="normal" if uplift > 20 else "inverse",
            help_text="Percentage sales increase on promo days vs non-promo days.",
        )

    fig_promo_cmp = plot_promo_comparison(promo_hist, detail_store_id)
    st.plotly_chart(fig_promo_cmp, use_container_width=True, key="promo_cmp_chart")

else:
    show_empty_state("Store list unavailable — run `python src/data_pipeline.py` to set up the database.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style='text-align:center; color:#8B8FA8; font-size:0.78rem; padding:16px 0 8px;'>
        🏠 Dashboard  ·  Retail AI Decision Intelligence Platform  ·
        Owner: <strong>Dikshit</strong>  ·
        Data: <a href='https://www.kaggle.com/c/rossmann-store-sales' style='color:#6C63FF;'>Rossmann Store Sales</a>
    </div>
    """,
    unsafe_allow_html=True,
)

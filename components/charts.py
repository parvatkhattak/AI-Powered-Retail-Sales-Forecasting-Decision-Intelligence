"""
components/charts.py
Owner: Dikshit — UI Developer

Reusable Plotly chart factory functions used across all pages.
All functions return a plotly Figure; rendering is done by the caller via st.plotly_chart().
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design Tokens ──────────────────────────────────────────────────────────────
PALETTE_PRIMARY = "#6C63FF"
PALETTE_ACCENT  = "#FF6584"
PALETTE_GREEN   = "#43D9A4"
PALETTE_ORANGE  = "#FFA552"
PALETTE_RED     = "#FF5370"
PALETTE_BG      = "#0E1117"
PALETTE_SURFACE = "#1A1D27"
PALETTE_BORDER  = "#2E3250"
PALETTE_TEXT    = "#E0E0FF"
PALETTE_MUTED   = "#8B8FA8"

STORE_TYPE_COLORS: dict[str, str] = {
    "a": PALETTE_PRIMARY,
    "b": PALETTE_GREEN,
    "c": PALETTE_ORANGE,
    "d": PALETTE_ACCENT,
}

_BASE_LAYOUT = dict(
    paper_bgcolor=PALETTE_SURFACE,
    plot_bgcolor=PALETTE_SURFACE,
    font=dict(family="Inter, sans-serif", color=PALETTE_TEXT, size=13),
    margin=dict(l=12, r=12, t=40, b=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE_BORDER),
    colorway=[PALETTE_PRIMARY, PALETTE_GREEN, PALETTE_ORANGE, PALETTE_ACCENT, PALETTE_RED],
)


def _apply_base(fig: go.Figure) -> go.Figure:
    """Apply shared dark-mode base layout to a figure."""
    fig.update_layout(**_BASE_LAYOUT)
    fig.update_xaxes(
        gridcolor=PALETTE_BORDER,
        showgrid=True,
        zeroline=False,
        tickfont=dict(color=PALETTE_MUTED),
    )
    fig.update_yaxes(
        gridcolor=PALETTE_BORDER,
        showgrid=True,
        zeroline=False,
        tickfont=dict(color=PALETTE_MUTED),
    )
    return fig


# ── 1. Sales Trend Line Chart ─────────────────────────────────────────────────

def plot_sales_trend(df: pd.DataFrame, store_id: int | None = None) -> go.Figure:
    """
    Line chart of daily/weekly/monthly sales over time.

    Args:
        df:       DataFrame with columns [period, total_sales, avg_sales] or
                  [store_id, date, sales, customers, promo].
        store_id: Optional label for chart title.

    Returns:
        plotly Figure.
    """
    title = f"Sales Trend — Store {store_id}" if store_id else "Fleet-Wide Sales Trend"

    if "period" in df.columns and "total_sales" in df.columns:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["period"],
                y=df["total_sales"],
                mode="lines+markers",
                name="Total Sales",
                line=dict(color=PALETTE_PRIMARY, width=2.5),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>Total Sales: €%{y:,.0f}<extra></extra>",
            )
        )
        if "avg_sales" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["period"],
                    y=df["avg_sales"],
                    mode="lines",
                    name="Avg Daily Sales",
                    line=dict(color=PALETTE_GREEN, width=1.8, dash="dot"),
                    hovertemplate="<b>%{x}</b><br>Avg Sales: €%{y:,.0f}<extra></extra>",
                )
            )
    else:
        # date-level data from get_store_metrics
        date_col = "date" if "date" in df.columns else df.columns[1]
        sales_col = "sales" if "sales" in df.columns else df.columns[2]
        fig = go.Figure()
        if "store_id" in df.columns and df["store_id"].nunique() > 1:
            for sid, grp in df.groupby("store_id"):
                fig.add_trace(
                    go.Scatter(
                        x=grp[date_col],
                        y=grp[sales_col],
                        mode="lines",
                        name=f"Store {sid}",
                        hovertemplate=f"Store {sid}<br><b>%{{x}}</b><br>€%{{y:,.0f}}<extra></extra>",
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[sales_col],
                    mode="lines+markers",
                    name="Daily Sales",
                    line=dict(color=PALETTE_PRIMARY, width=2.5),
                    marker=dict(size=5),
                    hovertemplate="<b>%{x}</b><br>€%{y:,.0f}<extra></extra>",
                )
            )
            if "promo" in df.columns:
                promo_days = df[df["promo"] == 1]
                fig.add_trace(
                    go.Scatter(
                        x=promo_days[date_col],
                        y=promo_days[sales_col],
                        mode="markers",
                        name="Promo Day",
                        marker=dict(color=PALETTE_ORANGE, size=9, symbol="star"),
                        hovertemplate="<b>Promo Day</b><br>%{x}<br>€%{y:,.0f}<extra></extra>",
                    )
                )

    fig.update_layout(title=dict(text=title, font=dict(size=16, color=PALETTE_TEXT)))
    return _apply_base(fig)


# ── 2. Store Ranking Bar Chart ────────────────────────────────────────────────

def plot_store_ranking(df: pd.DataFrame, top_n: int = 10, ascending: bool = False) -> go.Figure:
    """
    Horizontal bar chart of top/bottom stores by average sales.

    Args:
        df:         DataFrame with columns [Store, avg_sales] or [store_id, avg_sales].
        top_n:      How many stores to display.
        ascending:  False = top stores, True = bottom stores.

    Returns:
        plotly Figure.
    """
    store_col = "Store" if "Store" in df.columns else "store_id"
    sales_col = "avg_sales" if "avg_sales" in df.columns else "sales"

    df_sorted = df.sort_values(sales_col, ascending=ascending).head(top_n).copy()
    df_sorted[store_col] = df_sorted[store_col].astype(str).apply(lambda x: f"Store {x}")

    label = "Bottom" if ascending else "Top"
    color = PALETTE_RED if ascending else PALETTE_GREEN

    fig = go.Figure(
        go.Bar(
            x=df_sorted[sales_col],
            y=df_sorted[store_col],
            orientation="h",
            marker=dict(
                color=df_sorted[sales_col],
                colorscale=[[0, PALETTE_RED], [0.5, PALETTE_ORANGE], [1, PALETTE_GREEN]],
                showscale=False,
            ),
            hovertemplate="%{y}<br>Avg Sales: €%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{label} {top_n} Stores by Average Daily Sales",
            font=dict(size=16, color=PALETTE_TEXT),
        ),
        xaxis_title="Average Daily Sales (€)",
        yaxis=dict(categoryorder="total ascending" if ascending else "total descending"),
    )
    return _apply_base(fig)


# ── 3. Promo vs Non-Promo Comparison ─────────────────────────────────────────

def plot_promo_comparison(promo_data: dict, store_id: int) -> go.Figure:
    """
    Side-by-side bar comparing promo and non-promo average sales for a store.

    Args:
        promo_data: dict with keys [promo_avg_sales, non_promo_avg_sales, uplift_pct].
        store_id:   Store identifier for the title.

    Returns:
        plotly Figure.
    """
    categories = ["Non-Promo Days", "Promo Days"]
    values = [promo_data.get("non_promo_avg_sales", 0), promo_data.get("promo_avg_sales", 0)]
    colors = [PALETTE_MUTED, PALETTE_GREEN]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"€{v:,.0f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>€%{y:,.0f}<extra></extra>",
        )
    )
    uplift = promo_data.get("uplift_pct", 0)
    fig.update_layout(
        title=dict(
            text=f"Store {store_id} — Promo Impact  (+{uplift:.1f}% uplift)",
            font=dict(size=16, color=PALETTE_TEXT),
        ),
        yaxis_title="Average Daily Sales (€)",
    )
    return _apply_base(fig)


# ── 4. Promo Uplift Ranking ───────────────────────────────────────────────────

def plot_promo_uplift_ranking(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of stores ranked by promo uplift percentage.

    Args:
        df: DataFrame with columns [Store, uplift_pct, store_type] or similar.

    Returns:
        plotly Figure.
    """
    store_col = "Store" if "Store" in df.columns else "store_id"
    df = df.copy().sort_values("uplift_pct", ascending=True)
    df["label"] = df[store_col].astype(str).apply(lambda x: f"Store {x}")

    color_col = df.get("store_type", pd.Series(["a"] * len(df)))  # fallback

    fig = go.Figure(
        go.Bar(
            x=df["uplift_pct"],
            y=df["label"],
            orientation="h",
            marker=dict(
                color=df["uplift_pct"],
                colorscale=[[0, PALETTE_PRIMARY], [1, PALETTE_GREEN]],
                showscale=True,
                colorbar=dict(title="Uplift %", tickfont=dict(color=PALETTE_MUTED)),
            ),
            hovertemplate="%{y}<br>Uplift: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="Promo Uplift Ranking",
            font=dict(size=16, color=PALETTE_TEXT),
        ),
        xaxis_title="Promo Uplift (%)",
    )
    return _apply_base(fig)


# ── 5. Customer vs Sales Scatter ──────────────────────────────────────────────

def plot_customers_vs_sales(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of Customers vs Sales coloured by Promo flag.

    Args:
        df: DataFrame with columns [store_id, date, sales, customers, promo].

    Returns:
        plotly Figure.
    """
    if "customers" not in df.columns or "sales" not in df.columns:
        return go.Figure()

    df = df.copy()
    df["promo_label"] = df["promo"].map({1: "Promo Day", 0: "Regular Day"})
    color_map = {"Promo Day": PALETTE_ORANGE, "Regular Day": PALETTE_PRIMARY}

    fig = px.scatter(
        df,
        x="customers",
        y="sales",
        color="promo_label",
        color_discrete_map=color_map,
        hover_data={"date": True, "store_id": True},
        labels={"customers": "Customers", "sales": "Sales (€)", "promo_label": ""},
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        title=dict(
            text="Customers vs Sales — Promo vs Non-Promo",
            font=dict(size=16, color=PALETTE_TEXT),
        )
    )
    return _apply_base(fig)


# ── 6. KPI Sparkline ──────────────────────────────────────────────────────────

def plot_sparkline(values: list[float], color: str = PALETTE_PRIMARY) -> go.Figure:
    """
    Minimal sparkline chart for use inside metric cards.

    Args:
        values: List of numeric values to plot.
        color:  Line colour hex string.

    Returns:
        Compact plotly Figure.
    """
    fig = go.Figure(
        go.Scatter(
            y=values,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)",
        )
    )
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── 7. Store Type Breakdown ───────────────────────────────────────────────────

def plot_store_type_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: avg sales per store type.

    Args:
        df: DataFrame from get_all_stores() joined with per-store avg_sales.
            Expected columns: [StoreType or store_type, avg_sales].

    Returns:
        plotly Figure.
    """
    type_col = "StoreType" if "StoreType" in df.columns else "store_type"
    if type_col not in df.columns or "avg_sales" not in df.columns:
        return go.Figure()

    summary = df.groupby(type_col)["avg_sales"].agg(["mean", "count"]).reset_index()
    summary.columns = [type_col, "avg_sales", "store_count"]
    summary = summary.sort_values("avg_sales", ascending=False)

    fig = go.Figure()
    for _, row in summary.iterrows():
        stype = str(row[type_col])
        fig.add_trace(
            go.Bar(
                x=[f"Type {stype.upper()}"],
                y=[row["avg_sales"]],
                name=f"Type {stype.upper()} ({int(row['store_count'])} stores)",
                marker_color=STORE_TYPE_COLORS.get(stype, PALETTE_PRIMARY),
                hovertemplate=(
                    f"Store Type {stype.upper()}<br>"
                    f"Stores: {int(row['store_count'])}<br>"
                    "Avg Sales: €%{y:,.0f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(text="Average Sales by Store Type", font=dict(size=16, color=PALETTE_TEXT)),
        yaxis_title="Average Daily Sales (€)",
        barmode="group",
        showlegend=True,
    )
    return _apply_base(fig)


# ── 8. Anomaly Timeline ───────────────────────────────────────────────────────

def plot_anomaly_timeline(
    metrics_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    store_id: int,
) -> go.Figure:
    """
    Sales line chart with anomaly points highlighted as red markers.

    Args:
        metrics_df: Daily sales DataFrame [date, sales].
        anomaly_df: Anomaly rows [date, sales, sales_zscore].
        store_id:   Store label for title.

    Returns:
        plotly Figure.
    """
    fig = go.Figure()

    date_col  = "date"  if "date"  in metrics_df.columns else metrics_df.columns[0]
    sales_col = "sales" if "sales" in metrics_df.columns else metrics_df.columns[1]

    fig.add_trace(
        go.Scatter(
            x=metrics_df[date_col],
            y=metrics_df[sales_col],
            mode="lines",
            name="Daily Sales",
            line=dict(color=PALETTE_PRIMARY, width=2),
            hovertemplate="<b>%{x}</b><br>Sales: €%{y:,.0f}<extra></extra>",
        )
    )

    if not anomaly_df.empty:
        a_date_col  = "date"  if "date"  in anomaly_df.columns else anomaly_df.columns[0]
        a_sales_col = "sales" if "sales" in anomaly_df.columns else anomaly_df.columns[1]
        z_col       = "sales_zscore" if "sales_zscore" in anomaly_df.columns else None

        hover_text = (
            anomaly_df.apply(
                lambda r: f"<b>⚠ Anomaly</b><br>{r[a_date_col]}<br>Sales: €{r[a_sales_col]:,.0f}"
                          + (f"<br>Z-score: {r[z_col]:.2f}" if z_col else ""),
                axis=1,
            ).tolist()
            if z_col else None
        )

        fig.add_trace(
            go.Scatter(
                x=anomaly_df[a_date_col],
                y=anomaly_df[a_sales_col],
                mode="markers",
                name="Anomaly",
                marker=dict(color=PALETTE_RED, size=12, symbol="x-open", line=dict(width=2)),
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Store {store_id} — Sales with Anomaly Detection",
            font=dict(size=16, color=PALETTE_TEXT),
        ),
        xaxis_title="Date",
        yaxis_title="Sales (€)",
    )
    return _apply_base(fig)

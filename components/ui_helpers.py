"""
components/ui_helpers.py
Owner: Dikshit — UI Developer

Reusable Streamlit widget helpers used across all pages.
"""

import streamlit as st
import pandas as pd
from typing import Any


# ── Store Selector ─────────────────────────────────────────────────────────────

def store_selector(
    all_stores: pd.DataFrame,
    label: str = "Select Store",
    key: str = "store_selector",
) -> int:
    """
    Dropdown widget to pick a single store ID.

    Args:
        all_stores: DataFrame with a Store or store_id column.
        label:      Widget label shown to the user.
        key:        Unique Streamlit widget key.

    Returns:
        Selected store_id as int.
    """
    store_col = "Store" if "Store" in all_stores.columns else "store_id"
    options = sorted(all_stores[store_col].dropna().unique().tolist())
    selected = st.selectbox(label, options, format_func=lambda x: f"Store {x}", key=key)
    return int(selected)


# ── Multi-Store Selector ───────────────────────────────────────────────────────

def multi_store_selector(
    all_stores: pd.DataFrame,
    label: str = "Select Stores (max 5)",
    default_n: int = 3,
    key: str = "multi_store_selector",
) -> list[int]:
    """
    Multi-select widget for picking several stores.

    Args:
        all_stores: DataFrame with a Store or store_id column.
        label:      Widget label.
        default_n:  How many stores to pre-select.
        key:        Unique Streamlit widget key.

    Returns:
        List of selected store IDs.
    """
    store_col = "Store" if "Store" in all_stores.columns else "store_id"
    options = sorted(all_stores[store_col].dropna().unique().tolist())
    defaults = options[:default_n]
    selected = st.multiselect(
        label,
        options,
        default=defaults,
        format_func=lambda x: f"Store {x}",
        key=key,
        max_selections=5,
    )
    return [int(s) for s in selected]


# ── KPI Card ───────────────────────────────────────────────────────────────────

def kpi_card(
    label: str,
    value: Any,
    delta: str | None = None,
    delta_color: str = "normal",
    help_text: str | None = None,
) -> None:
    """
    Render a single KPI metric tile using st.metric.

    Args:
        label:       The KPI name shown above the value.
        value:       The main value to display.
        delta:       Optional delta string (e.g. '+5.2%').
        delta_color: 'normal' | 'inverse' | 'off'.
        help_text:   Optional tooltip text.
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color, help=help_text)


# ── Citation Card ──────────────────────────────────────────────────────────────

def citation_card(sources: list[str]) -> None:
    """
    Render a styled info box listing data sources that backed an AI response.

    Args:
        sources: List of source strings, e.g. ['database.get_store_metrics', 'model_engine.get_7day_forecast'].
    """
    if not sources:
        return
    lines = "\n".join(f"- `{s}`" for s in sources)
    st.info(f"**📚 Data Sources**\n\n{lines}", icon="📌")


# ── Download CSV Button ────────────────────────────────────────────────────────

def download_csv_button(df: pd.DataFrame, filename: str = "data.csv") -> None:
    """
    Render a download button that exports a DataFrame as CSV.

    Args:
        df:       The DataFrame to export.
        filename: Suggested filename for the downloaded file.
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


# ── Section Header ─────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "") -> None:
    """
    Render a styled section header with optional subtitle.

    Args:
        title:    Section title (h3 level).
        subtitle: Optional short description.
    """
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


# ── Empty / Error State ────────────────────────────────────────────────────────

def show_empty_state(message: str = "No data available for the selected filters.") -> None:
    """
    Render a user-friendly empty state message instead of a raw error.

    Args:
        message: The message to display.
    """
    st.warning(f"⚠️ {message}", icon="📭")


def show_error_state(error_msg: str) -> None:
    """
    Render a user-friendly error state without exposing raw Python exceptions.

    Args:
        error_msg: Human-readable description of what went wrong.
    """
    st.error(f"❌ {error_msg}", icon="🔥")

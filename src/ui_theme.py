"""
src/ui_theme.py
Shared app-wide look: the hero background illustration + dark theme, applied
identically on every page. Call apply_theme() once near the top of each page
script (after st.set_page_config), right after the imports.

The background image lives at assests/hero_bg.png — a cropped, top-faded copy
of the reference illustration (title/card UI baked into the original screenshot
is cropped out, since every page already builds its own real title/cards).
"""
import base64
from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assests"
_ASSET_PATH = _ASSETS_DIR / "hero_bg.png"


@st.cache_data(show_spinner=False)
def _hero_bg_b64(mtime: float) -> str:
    # mtime busts the cache automatically if the asset is ever replaced.
    return base64.b64encode(_ASSET_PATH.read_bytes()).decode()


@st.cache_data(show_spinner=False)
def _asset_b64(path_str: str, mtime: float) -> str:
    return base64.b64encode(Path(path_str).read_bytes()).decode()


def asset_data_uri(filename: str, mime: str = "image/png") -> str | None:
    """Returns a data: URI for a file in assests/, for embedding in <img src=...>
    inside components.html (which can't reach local file paths directly).
    Returns None if the file doesn't exist, so callers can fall back cleanly."""
    path = _ASSETS_DIR / filename
    if not path.exists():
        return None
    b64 = _asset_b64(str(path), path.stat().st_mtime)
    return f"data:{mime};base64,{b64}"


def apply_theme():
    """Injects the shared background + dark theme. Safe to call on every page."""
    if not _ASSET_PATH.exists():
        b64 = None
    else:
        b64 = _hero_bg_b64(_ASSET_PATH.stat().st_mtime)

    bg_layers = (
        f'url("data:image/png;base64,{b64}")' if b64 else "none"
    )

    st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-color: #05130c !important;
            background-image:
                linear-gradient(180deg, rgba(2,12,8,0.55) 0%, rgba(2,12,8,0.7) 60%, rgba(0,0,0,0.97) 100%),
                {bg_layers} !important;
            /* The image is a fixed-width band anchored to the bottom, not a full-bleed
               cover — otherwise it stretches to fill short pages and washes out text. */
            background-size: cover, 100% auto !important;
            background-position: center, center bottom !important;
            background-repeat: no-repeat, no-repeat !important;
            background-attachment: fixed, fixed !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0) !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #050b08 0%, #000000 100%) !important;
            border-right: 1px solid rgba(16,185,129,0.15);
        }}
        [data-testid="stSidebar"] * {{ color: #e5e7eb !important; }}
        [data-testid="stSidebarNav"] a {{ border-radius: 8px; }}
        [data-testid="stSidebarNav"] a:hover {{ background: rgba(16,185,129,0.12) !important; }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(16,185,129,0.18) !important;
            color: #6ee7b7 !important;
        }}

        /* Default markdown/heading/caption text — light, readable on the dark background */
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {{
            color: #e5e7eb !important;
        }}
        .stMarkdown table {{ color: #e5e7eb !important; border-color: rgba(255,255,255,0.15) !important; }}
        .stMarkdown th, .stMarkdown td {{ border-color: rgba(255,255,255,0.15) !important; }}
    </style>
    """, unsafe_allow_html=True)

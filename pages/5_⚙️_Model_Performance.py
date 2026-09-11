import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui_theme import apply_theme

st.set_page_config(page_title="Model Performance", page_icon="⚙️", layout="wide")
apply_theme()

st.title("⚙️ Model Performance")
st.info("MOCK: Model Performance page stub. Dikshit will implement RMSPE metrics and SHAP beeswarm here.")

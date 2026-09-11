import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui_theme import apply_theme

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
apply_theme()

st.title("🤖 AI Assistant")
st.info("MOCK: AI Assistant page stub. Dikshit will implement the streaming chat interface here.")

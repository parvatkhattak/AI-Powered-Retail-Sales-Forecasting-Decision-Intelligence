import streamlit as st
from config import APP_TITLE, APP_ICON, APP_LAYOUT

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.title("Retail AI")
    st.caption("Sales Forecasting & Decision Intelligence")
    st.divider()
    st.markdown(
        """
        **Navigate:**
        - 🏠 Dashboard
        - 📈 Forecasting
        - 🔍 Promotion Analysis
        - 🤖 AI Assistant
        - ⚙️ Model Performance
        """
    )
    st.divider()
    st.caption("Dataset: Rossmann Store Sales")
    st.caption("Model: XGBoost + LightGBM")
    st.caption("Agent: LangGraph + LLM")

# ── Landing page ───────────────────────────────────────────────────────────────
st.title("🛒 Retail AI — Decision Intelligence Platform")
st.markdown(
    """
    Welcome to the **AI-Powered Retail Sales Forecasting & Decision Intelligence** platform.

    Use the sidebar to navigate between pages:

    | Page | What You Can Do |
    |------|----------------|
    | 🏠 **Dashboard** | Fleet-wide KPIs, sales trends, store rankings, anomaly detection |
    | 📈 **Forecasting** | 7-day store-level sales forecast with SHAP explanations |
    | 🔍 **Promotion Analysis** | Promo uplift ranking, store type breakdown |
    | 🤖 **AI Assistant** | Ask natural language questions about any store |
    | ⚙️ **Model Performance** | RMSPE metrics, walk-forward validation, baseline comparison |
    """
)

st.info(
    "💡 **Try the AI Assistant** — Ask: "
    "*'I manage Stores 100, 200, 300, 400, 500. Which should I focus on next week?'*",
    icon="🤖"
)

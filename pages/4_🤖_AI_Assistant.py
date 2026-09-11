"""
pages/4_🤖_AI_Assistant.py
Owner: Dikshit — UI Developer

Streaming chat interface powered by src/agent_graph.py.

Features:
  1. Streaming chat using run_agent_stream() with real-time token display
  2. Session message history (persists across re-runs via st.session_state)
  3. Clickable example question chips
  4. Citation cards showing which data sources backed each response
  5. Graceful error handling — agent failures show friendly message

Architecture constraints:
  - Calls ONLY run_agent_stream() and run_agent() from src/agent_graph.py
  - Never imports from database.py or model_engine.py directly
  - All errors wrapped in try/except — never crashes Streamlit
"""

import logging
import sys
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import USE_MOCKS
from src.agent_graph import run_agent_stream
from components.ui_helpers import citation_card

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant — Retail AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0c14 0%, #0e1117 50%, #0a0f1e 100%); }
    hr { border-color: #2E3250 !important; margin: 2rem 0 !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #141727 100%);
        border-right: 1px solid #2E3250;
    }
    .page-title {
        background: linear-gradient(90deg, #43D9A4, #6C63FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 700; margin-bottom: 0.1rem;
    }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
             font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }
    .badge-live { background: rgba(67,217,164,0.15); color: #43D9A4; border: 1px solid #43D9A4; }
    .badge-mock { background: rgba(255,165,82,0.15); color: #FFA552; border: 1px solid #FFA552; }

    /* Chat message bubbles */
    div[data-testid="stChatMessage"] {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2133 100%);
        border: 1px solid #2E3250;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    /* Chat input box */
    div[data-testid="stChatInput"] > div {
        background: #1a1d27;
        border: 1px solid #2E3250;
        border-radius: 12px;
    }
    /* Example chips */
    .chip-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 20px; }
    .chip {
        background: rgba(108,99,255,0.10);
        border: 1px solid #6C63FF;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #C0BCFF;
        cursor: pointer;
        transition: background 0.2s;
    }
    .chip:hover { background: rgba(108,99,255,0.25); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""
if "awaiting_answer" not in st.session_state:
    st.session_state["awaiting_answer"] = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=52)
    st.markdown("## 🤖 AI Assistant")
    st.divider()

    mode_badge = (
        '<span class="badge badge-mock">🟡 MOCK DATA</span>'
        if USE_MOCKS
        else '<span class="badge badge-live">🟢 LIVE DATA</span>'
    )
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.divider()

    st.markdown(
        """
        **How to use:**
        1. Type a question or click an example chip below
        2. The assistant queries real store data
        3. Citations show which data sources were used

        **Supported query types:**
        - 🔍 *Performance* — "How is Store 125 doing?"
        - 📈 *Forecast* — "What are next week's sales for Store 200?"
        - 🎯 *Recommend* — "Which of stores 100–500 should I focus on?"
        - 🔮 *What-If* — "What if Store 300 runs a promo next week?"
        """
    )
    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat"):
        st.session_state["messages"] = []
        st.session_state["pending_query"] = ""
        st.session_state["awaiting_answer"] = ""
        st.rerun()

    st.caption("📦 Dataset: Rossmann Store Sales")
    st.caption("🏗️ Owner: Dikshit (pages/)")

# ── Page Header ───────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="page-title">AI Retail Assistant</p>', unsafe_allow_html=True)
    st.markdown("Ask natural-language questions about any store. Get grounded, evidence-backed answers.")
with col_badge:
    st.markdown(
        f"<div style='text-align:right;padding-top:12px;'>{mode_badge}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Example Question Chips ────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "I manage Stores 100, 200, 300, 400, 500. Which should I focus on next week?",
    "How is Store 125 performing over the last 30 days?",
    "What would happen to Store 200's sales if we added a promotion next week?",
    "Which stores have the highest promo uplift?",
    "What are the expected sales for Store 300 next week?",
    "Compare Store 50 and Store 200 — which is doing better?",
]

st.markdown("##### 💡 Try an example question:")
chip_cols = st.columns(3)
for idx, question in enumerate(EXAMPLE_QUESTIONS):
    col_idx = idx % 3
    with chip_cols[col_idx]:
        if st.button(
            f"💬 {question[:55]}{'…' if len(question) > 55 else ''}",
            key=f"chip_{idx}",
            use_container_width=True,
            help=question,
        ):
            st.session_state["pending_query"] = question

st.divider()

# ── Chat History Display ──────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

        # Show citation card for assistant messages that have sources
        if msg["role"] == "assistant" and msg.get("sources"):
            citation_card(msg["sources"])

# ── Chat Input ────────────────────────────────────────────────────────────────
# st.chat_input is rendered unconditionally. It used to sit on the right-hand
# side of an `or`, so clicking an example chip short-circuited it away and the
# input box vanished from the page for that run.
typed_query = st.chat_input(
    "Ask anything about your stores… e.g. 'Which store should I focus on next week?'",
    key="chat_input",
)
prompt = st.session_state.pop("pending_query", "") or typed_query

if prompt:
    # The user's message is committed to history and the script re-runs before
    # the (slow) agent call starts. Previously both messages were appended only
    # after the call returned, so submitting a second question mid-answer — or
    # any failure during it — discarded the exchange and the visible text
    # disappeared.
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["awaiting_answer"] = prompt
    st.rerun()

# ── Generate the pending answer ───────────────────────────────────────────────
if st.session_state.get("awaiting_answer"):
    question = st.session_state.pop("awaiting_answer")

    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""
        sources: list[str] = []

        try:
            with st.spinner("Analysing your stores…"):
                for chunk in run_agent_stream(question, session_id="streamlit_session"):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Citations are only ever what the agent actually reported. There
            # used to be a fallback that invented two source names when none
            # were parsed, which attributed data to functions that may never
            # have run.
            if "Sources:" in full_response:
                tail = full_response.split("Sources:")[-1]
                sources = [
                    line.strip("- •*").strip()
                    for line in tail.strip().splitlines()
                    if line.strip("- •*").strip()
                ]

            if sources:
                citation_card(sources)

        except Exception:
            # The detail goes to the server log, not to the user: exception
            # text can carry SQL, file paths and schema details.
            logger.exception("AI Assistant failed to answer: %r", question)
            full_response = (
                "⚠️ I couldn't process that request. Please try asking about store "
                "sales, forecasts, promotions, or which stores need attention."
            )
            response_placeholder.markdown(full_response)

        finally:
            # finally, not the try body: Streamlit raises a BaseException to
            # stop the script when the user submits again mid-answer, so this
            # is what guarantees the exchange is still saved.
            st.session_state["messages"].append({
                "role": "assistant",
                "content": full_response or "⚠️ That answer didn't finish. Please ask again.",
                "sources": sources,
            })

# ── Empty state when no messages yet ─────────────────────────────────────────
if not st.session_state["messages"]:
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 60px 20px;
            color: #8B8FA8;
            border: 1px dashed #2E3250;
            border-radius: 16px;
            margin: 20px 0;
        ">
            <div style="font-size: 3rem; margin-bottom: 16px;">🤖</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #C0BCFF; margin-bottom: 8px;">
                Your AI Retail Assistant is ready
            </div>
            <div style="font-size: 0.9rem;">
                Click an example chip above or type your own question below.<br>
                All responses are grounded in real store data — no hallucinations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; color:#8B8FA8; font-size:0.78rem; padding:16px 0 8px;'>
        🤖 AI Assistant &nbsp;·&nbsp; Retail AI Decision Intelligence Platform &nbsp;·&nbsp;
        Owner: <strong>Dikshit</strong> &nbsp;·&nbsp;
        Powered by: <strong>LangGraph + LLM</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

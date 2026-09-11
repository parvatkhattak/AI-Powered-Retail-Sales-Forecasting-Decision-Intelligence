"""
pages/4_🤖_AI_Assistant.py
Owner: Dikshit — Frontend Developer / Saumya — AI Engineer

AI Assistant page:
- Streaming chat interface powered by LangGraph agent
- Citation cards showing source tools used
- Example question chips to guide users
- Full conversation history in session
- Loading spinner while agent thinks
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui_theme import apply_theme

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
apply_theme()

st.markdown("""
<style>
    /* User message bubble */
    .user-bubble {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        color: white;
        font-size: 1.05rem;
        max-width: 80%;
        margin-left: auto;
        display: block;
        word-wrap: break-word;
    }
    /* Agent message bubble */
    .agent-bubble {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        color: #e5e7eb;
        font-size: 1.05rem;
        max-width: 85%;
        word-wrap: break-word;
    }
    /* Citation card */
    .citation-card {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 8px;
        padding: 0.25rem 0.6rem;
        margin: 0.2rem 0.2rem 0 0;
        font-size: 0.78rem;
        color: #93c5fd;
        font-family: monospace;
    }
    /* Chip buttons */
    div[data-testid="stHorizontalBlock"] button {
        font-size: 0.88rem !important;
        padding: 0.35rem 0.8rem !important;
        border-radius: 999px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 AI Decision Assistant")
st.markdown("**Ask natural-language questions about any store's performance, forecasts, and promotions.**")
st.divider()

# ── Session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"assistant", "text": str, "citations": list}

# ── Example question chips ─────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What is the forecasted sales for Store 1 next week?",
    "Should Store 5 run a promotion next week?",
    "I manage Stores 100, 200, 300, 400, 500. Which needs the most attention?",
    "What drives Store 250's sales the most?",
    "What would happen to Store 10's sales if we add a promotion?",
]

st.markdown("**💡 Try an example question:**")
chips = st.columns(len(EXAMPLE_QUESTIONS))
prefill_question = None
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if chips[i].button(q[:40] + "…" if len(q) > 40 else q, key=f"chip_{i}", use_container_width=True):
        prefill_question = q

# ── Chat input ─────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask a question about any store:",
        value=prefill_question or "",
        placeholder="e.g. What is the 7-day forecast for Store 1?",
        label_visibility="collapsed",
    )
    col_send, col_clear = st.columns([4, 1])
    submitted = col_send.form_submit_button("Send →", use_container_width=True)
    cleared = col_clear.form_submit_button("Clear Chat", use_container_width=True)

if cleared:
    st.session_state.chat_history = []
    st.rerun()

# ── Handle submission ──────────────────────────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "text": question, "citations": []})

    # Show spinner while agent works
    with st.spinner("🤖 Agent is thinking…"):
        try:
            from src.agent_graph import run_agent
            raw_response = run_agent(question)

            # Parse citations from response — format: "📚 **Sources:** tool1, tool2"
            citations = []
            response_text = raw_response
            if "**Sources:**" in raw_response:
                parts = raw_response.split("**Sources:**")
                response_text = parts[0].strip()
                citation_line = parts[1].strip() if len(parts) > 1 else ""
                citations = [c.strip() for c in citation_line.replace("📚", "").split(",") if c.strip()]
            elif "Sources:" in raw_response:
                parts = raw_response.split("Sources:")
                response_text = parts[0].strip()
                citation_line = parts[1].strip() if len(parts) > 1 else ""
                citations = [c.strip() for c in citation_line.split(",") if c.strip()]

            st.session_state.chat_history.append({
                "role": "assistant",
                "text": response_text,
                "citations": citations,
            })
        except Exception as e:
            error_msg = (
                f"⚠️ Agent encountered an error: `{e}`\n\n"
                "Make sure `retail.db` is initialised and `OPENROUTER_API_KEY` is set in `.env`."
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "text": error_msg,
                "citations": [],
            })

    st.rerun()

# ── Render conversation ────────────────────────────────────────────────────────
st.divider()

if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; opacity: 0.5;">
        <div style="font-size: 4rem;">🤖</div>
        <p style="font-size: 1.2rem; color: #9ca3af;">
            Ask a question above to start a conversation with the AI assistant.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Render agent bubble
            st.markdown(
                f'<div class="agent-bubble">🤖 {msg["text"]}</div>',
                unsafe_allow_html=True,
            )
            # Render citation cards
            if msg.get("citations"):
                citation_html = " ".join(
                    f'<span class="citation-card">📎 {c}</span>'
                    for c in msg["citations"]
                )
                st.markdown(
                    f'<div style="margin-top:0.3rem; margin-bottom:0.8rem;">{citation_html}</div>',
                    unsafe_allow_html=True,
                )

st.divider()
st.caption("⚠️ The AI assistant only uses real data from the database and model. It never invents numbers.")

"""
diagnose_latency.py
Owner: Saumya — Agentic AI Engineer

Splits a query's wall-clock time into the part this codebase controls (the
database, the model, the graph) and the part the LLM provider controls.

Run it when answers feel slow:

    python diagnose_latency.py

It reports, per question: time inside the agent's own tools, time waiting on
OpenRouter, the input and output token counts, and the resulting tokens/second.
That last number is the one that decides what to do — a slow model and a slow
pipeline need opposite fixes, and guessing between them wastes a day.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LLM_MAX_TOKENS, LLM_MODEL
from src import agent_graph

QUESTIONS = [
    "How is Store 250 performing?",
    "What are expected sales for Store 300 next week?",
    "What if Store 300 runs a promo next week?",
    "Compare Stores 100, 200 and 300 and recommend one action.",
    "Which 5 stores are most at risk of underperforming next week?",
]

# Rough, and honestly labelled as such: good enough to tell 300 tokens from
# 3,000, which is the only distinction that changes a decision here.
CHARS_PER_TOKEN = 4


def _time_raw_llm_call() -> float | None:
    """One trivial round-trip, to separate provider overhead from generation."""
    from langchain_core.messages import HumanMessage

    try:
        started = time.perf_counter()
        agent_graph._get_llm().invoke([HumanMessage(content="Reply with the single word: ok")])
        return time.perf_counter() - started
    except Exception as exc:
        print(f"  ! raw LLM call failed: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    status = agent_graph.check_llm_connection()
    print(f"Model      : {LLM_MODEL}")
    print(f"Max tokens : {LLM_MAX_TOKENS}")
    print(f"Reachable  : {status['ok']}  ({status['detail']})\n")

    if status["ok"]:
        overhead = _time_raw_llm_call()
        if overhead is not None:
            print(f"Round-trip for a one-word reply: {overhead:.2f}s")
            print("  (that is pure provider latency — queueing, cold start, network;\n"
                  "   no code change in this repo can reduce it)\n")

    print(f"{'question':<52}{'tools':>8}{'llm':>8}{'total':>8}{'in':>7}{'out':>7}{'tok/s':>8}")
    print("-" * 98)

    for question in QUESTIONS:
        # The graph runs every tool but composition is stubbed out, so this is
        # the pipeline's own cost with the model removed.
        import json

        captured: dict = {}
        real_compose = agent_graph._compose_with_llm

        def measure(state, context, _real=real_compose):
            captured["prompt_chars"] = len(json.dumps(context, default=str))
            started = time.perf_counter()
            try:
                reply = _real(state, context)
            finally:
                captured["llm_seconds"] = time.perf_counter() - started
            captured["reply_chars"] = len(reply or "")
            return reply

        agent_graph._compose_with_llm = measure
        try:
            started = time.perf_counter()
            agent_graph.run_agent(question, f"diag-{question[:12]}")
            total = time.perf_counter() - started
        finally:
            agent_graph._compose_with_llm = real_compose

        llm_seconds = captured.get("llm_seconds", 0.0)
        tools = total - llm_seconds
        tokens_in = captured.get("prompt_chars", 0) // CHARS_PER_TOKEN
        tokens_out = captured.get("reply_chars", 0) // CHARS_PER_TOKEN
        rate = tokens_out / llm_seconds if llm_seconds > 0.05 and tokens_out else 0

        print(f"{question[:50]:<52}{tools:>7.2f}s{llm_seconds:>7.2f}s{total:>7.2f}s"
              f"{tokens_in:>7}{tokens_out:>7}{rate:>8.1f}")

    print("\nHow to read this:")
    print("  llm column dominates      -> provider latency. Switch to a faster model,")
    print("                               or lower LLM_MAX_TOKENS in .env.")
    print("  tools column dominates    -> this codebase. Send me the numbers.")
    print("  tok/s below ~15           -> the free tier is queueing you.")
    print("  out tokens near the cap   -> answers are being padded; lower LLM_MAX_TOKENS.")


if __name__ == "__main__":
    main()

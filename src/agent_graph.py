"""
src/agent_graph.py
Owner: Saumya — Agentic AI Engineer

Responsibilities:
- Define LangGraph state and nodes
- Route user queries
- Execute tool calls
- Format final responses
"""

from typing import Generator

def run_agent(user_query: str, session_id: str = "default") -> str:
    """Synchronous agent call."""
    # TODO (Saumya): Implement full LangGraph execution
    return f"MOCK: Here is a simulated response for query: '{user_query}'"

def run_agent_stream(user_query: str, session_id: str = "default") -> Generator[str, None, None]:
    """Streaming version for Streamlit chat UI."""
    # TODO (Saumya): Implement streaming LangGraph execution
    mock_chunks = ["Observation: ", "Store 200 is down 18%. ", "\nRecommendation: ", "Run Promo 1."]
    for chunk in mock_chunks:
        yield chunk

# Retail AI Capstone — Team Coding Guidelines
# This file is read by GitHub Copilot and enforces consistent standards across all 5 developers.

## Tech Stack
- Python 3.11
- Streamlit (UI)
- Pandas + NumPy (data manipulation)
- SQLite via SQLAlchemy (database)
- XGBoost + LightGBM (ML models)
- SHAP (explainability)
- LangGraph (agent framework)
- OpenRouter (LLM access)
- Plotly (charts)
- pytest (testing)

## Module Ownership — Do Not Edit Other Devs' Core Files Without Permission
- src/data_pipeline.py → Himanshu
- src/database.py → Himanshu
- src/model_engine.py → Ashutosh
- src/feature_engineering.py → Ashutosh
- src/agent_graph.py → Saumya
- src/decision_engine.py → Saumya
- src/prompts.py → Saumya
- pages/ → Dikshit
- components/ → Dikshit
- app.py → Parvat
- config.py → Parvat

## Code Style
- Use strict Python type hints on all function parameters and return types
- All functions must have a docstring (one-liner minimum)
- Use snake_case for variables/functions, PascalCase for classes
- Max line length: 100 characters
- Use f-strings for string formatting (no % or .format())

## Architecture Rules
- NEVER query SQLite directly from app.py or pages/ — use database.py functions only
- NEVER load .pkl files outside model_engine.py — use model_engine.py functions only
- NEVER hardcode store IDs — always parameterize
- NEVER hardcode file paths — always use constants from config.py
- The LLM must NEVER generate numbers — all figures must come from tool call return values

## Database
- All SQLite access through src/database.py using SQLAlchemy
- Use parameterised queries only (never f-strings in SQL)
- Index on: store_id, date, (store_id, date)

## Agent Structure
- All LangGraph nodes must accept and return AgentState TypedDict
- All tool functions must validate their inputs before calling DB or model
- Every agent response must include a data_sources list for citation cards

## Streamlit UI
- Use st.cache_data for data-fetching functions (TTL = 3600 seconds)
- Use st.cache_resource for model/DB connection loading
- Use streamlit-native components: st.dataframe, st.plotly_chart, st.chat_message
- All pages must handle empty/error states gracefully — no raw Python exceptions shown to user

## Testing
- Each src/ module has a corresponding tests/test_<module>.py
- Use pytest fixtures from tests/conftest.py for DB connection and model loading
- No test should take more than 10 seconds

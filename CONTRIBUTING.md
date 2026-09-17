# 🤝 Contributing to Retail AI — Sales Forecasting & Decision Intelligence

Welcome! This document explains how to contribute to the project, including the detailed guide for adding new store types.

---

## 📋 Table of Contents

- [Project Structure](#-project-structure)
- [Development Setup](#-development-setup)
- [Branching Strategy](#-branching-strategy)
- [How to Add a New Store Type](#-how-to-add-a-new-store-type)
- [Writing Tests](#-writing-tests)
- [Code Style](#-code-style)
- [Pull Request Checklist](#-pull-request-checklist)

---

## 📁 Project Structure

```
capstone/
├── app.py                    # Streamlit entry point
├── config.py                 # All paths and feature flags
├── src/
│   ├── data_pipeline.py      # ETL — cleans CSV, writes retail.db
│   ├── database.py           # All SQL query functions (single source of truth)
│   ├── feature_engineering.py# Lag features, rolling stats, promo flags
│   ├── model_engine.py       # Forecast, SHAP, What-If functions
│   ├── agent_graph.py        # LangGraph AI agent
│   └── decision_engine.py    # Structured Observation/Prediction/Evidence/Recommendation
├── pages/
│   ├── 1_🏠_Dashboard.py
│   ├── 2_📈_Forecasting.py
│   ├── 3_🔍_Promotion_Analysis.py
│   ├── 4_🤖_AI_Assistant.py
│   └── 5_⚙️_Model_Performance.py
├── components/               # Shared charts and UI helpers
├── tests/                    # pytest test suites
├── data/                     # CSVs + retail.db (not committed)
└── models/                   # Trained .pkl files
```

---

## 🛠 Development Setup

```bash
# 1. Clone the repo
git clone https://github.com/parvatkhattak/AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence.git
cd AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in your API key
cp .env.example .env     # then edit .env and set OPENROUTER_API_KEY

# 4. Build the database (needs train.csv and store.csv in data/)
python src/data_pipeline.py

# 5. Train the models
python compare_models.py

# 6. Run the app
streamlit run app.py
```

**Or use Docker (one command):**

```bash
docker compose up --build
# App is live at http://localhost:8501
```

---

## 🌿 Branching Strategy

```
main              ← production-ready code; all merges reviewed by Parvat
feature/<name>    ← your feature branch
fix/<name>        ← bug fix branches
```

Always branch off `main` and open a Pull Request when ready.

---

## 🏪 How to Add a New Store Type

The Rossmann dataset has store types `a`, `b`, `c`, `d`. If you receive data with a new type (e.g. `e` — a new flagship format), follow these steps exactly. Each step references the file you need to change.

---

### Step 1 — Verify the raw data

```python
import pandas as pd
df = pd.read_csv("data/store.csv")
print(df["StoreType"].unique())   # should include your new type
```

Make sure your `train.csv` and `store.csv` already contain rows for the new type.

---

### Step 2 — Re-run the data pipeline

```bash
python src/data_pipeline.py
```

[`src/data_pipeline.py`](src/data_pipeline.py) reads `store.csv`, merges it with `train.csv`, and writes `data/retail.db`. The `StoreType` column is stored as-is — no hardcoding — so the new type will appear automatically.

> **Check:** `SELECT DISTINCT StoreType FROM sales;` in `data/retail.db` should now include your new type.

---

### Step 3 — Update the chart colour palette

Open [`components/charts.py`](components/charts.py) and find the `STORE_TYPE_COLORS` dict:

```python
# Before
STORE_TYPE_COLORS = {
    "a": PALETTE_PRIMARY,   # blue
    "b": PALETTE_GREEN,     # green
    "c": PALETTE_ORANGE,    # orange
    "d": PALETTE_RED,       # red
}

# After — add your new type
STORE_TYPE_COLORS = {
    "a": PALETTE_PRIMARY,
    "b": PALETTE_GREEN,
    "c": PALETTE_ORANGE,
    "d": PALETTE_RED,
    "e": PALETTE_ACCENT,    # ← new type gets a colour
}
```

All charts that call `STORE_TYPE_COLORS.get(stype, PALETTE_PRIMARY)` will now use the right colour automatically.

---

### Step 4 — Update the AI prompt (so the assistant knows about it)

Open [`src/prompts.py`](src/prompts.py) and find the section that describes store types:

```python
# Before
STORE_TYPE_DESCRIPTIONS = {
    "a": "standard mid-size store",
    "b": "large flagship store",
    "c": "small convenience store",
    "d": "attached to a mall or shopping centre",
}

# After
STORE_TYPE_DESCRIPTIONS = {
    "a": "standard mid-size store",
    "b": "large flagship store",
    "c": "small convenience store",
    "d": "attached to a mall or shopping centre",
    "e": "new flagship format — opened after 2015",   # ← add description
}
```

The AI agent uses this dictionary when explaining store-level analysis. Without it, it will say `"unknown type"`.

---

### Step 5 — Retrain the model

The `StoreType` column is one-hot-encoded during feature engineering. A new type means new dummy columns.

```bash
# Retrain and save new .pkl files
python compare_models.py
```

Open [`src/feature_engineering.py`](src/feature_engineering.py) and confirm `StoreType` is in the `categorical_columns` list:

```python
categorical_columns = ["StoreType", "Assortment", "StateHoliday"]
```

If it is, `pd.get_dummies()` will handle the new type automatically and produce a `StoreType_e` column. The model retraining step handles the rest.

> ⚠️ **Important:** Commit the new `.pkl` files (`models/lgbm_model.pkl`, `models/xgboost_model.pkl`, etc.) together with the code change. A version mismatch between code and model file will cause silent failures.

---

### Step 6 — Update the mock data (for offline testing)

If `USE_MOCKS = True` in [`config.py`](config.py), update the mock file to include your new type:

```json
// data/mocks/mock_store_metrics.json  — add a few stores of type "e"
[
  { "store_id": 9001, "store_type": "e", "sales": 18500, "date": "2015-07-01" },
  ...
]
```

---

### Step 7 — Add a test

Open [`tests/test_database.py`](tests/test_database.py) and add an assertion:

```python
def test_new_store_type_e_present():
    from src.database import get_all_stores
    df = get_all_stores()
    assert "e" in df["StoreType"].values, "New store type 'e' not found in database"
```

Run all tests to confirm nothing broke:

```bash
pytest tests/ -q
```

---

### Step 8 — Open a Pull Request

1. Commit all changed files: `Dockerfile`, `components/charts.py`, `src/prompts.py`, `src/feature_engineering.py`, model `.pkl` files, mock JSON, and your new test.
2. Open a PR against `main` with the title: `feat: add StoreType e support`
3. Tag **Parvat** as reviewer — model file changes need lead sign-off.

---

## 🧪 Writing Tests

All tests live in [`tests/`](tests/). Each test file maps to one module:

| Test File | Module It Tests |
|-----------|----------------|
| `test_data_pipeline.py` | `src/data_pipeline.py` |
| `test_database.py` | `src/database.py` |
| `test_model_engine.py` | `src/model_engine.py` |
| `test_agent_graph.py` | `src/agent_graph.py` |
| `test_decision_engine.py` | `src/decision_engine.py` |
| `test_curveball.py` | End-to-end curveball scenario |

Run all tests:

```bash
pytest tests/ -q
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🎨 Code Style

- **Python:** Follow PEP 8. Use f-strings, not `.format()`.
- **SQL:** All SQL queries must live in `src/database.py` only — never write SQL directly in page files.
- **ML:** All model calls must go through `src/model_engine.py` — never load `.pkl` files directly in pages.
- **Imports:** Use `from config import ...` for all paths — never hardcode file paths.
- **Error handling:** Wrap all data fetches in `try/except` — never let a data error crash Streamlit.

---

## ✅ Pull Request Checklist

Before submitting a PR, confirm:

- [ ] `pytest tests/ -q` passes with zero failures
- [ ] No SQL written directly in `pages/` or `app.py`
- [ ] No hardcoded store IDs (e.g. `store_id = 100`)
- [ ] All new file paths imported from `config.py`
- [ ] If you changed a shared function, you informed the team first
- [ ] If you retrained models, new `.pkl` files are committed
- [ ] `README.md` updated if you added a new feature or page

---

*Questions? Message Parvat on the team channel or open a GitHub Discussion.*

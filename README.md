# 🛒 Retail AI — Sales Forecasting & Decision Intelligence

> **An AI-powered platform that helps retail store managers analyze performance, predict future sales, and receive evidence-backed recommendations — all through a natural language interface.**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Primary_Model-green)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-purple)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-288_passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Live Demo Screenshot](#-live-demo-screenshot)
- [Team](#-team)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start — Docker](#-quick-start--docker-one-command)
- [Quick Start — Local](#-quick-start--local)
- [Application Pages](#-application-pages)
- [How the AI Works](#-how-the-ai-works)
- [Dataset](#-dataset)
- [Running Tests](#-running-tests)
- [The Curveball Question](#-the-curveball-question)
- [Known Limitations](#-known-limitations)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## 🎯 What This Project Does

This platform answers real business questions for a retail manager who operates 1,000+ stores:

| Question | How We Answer It |
|----------|--------------------|
| *"How is Store 125 performing?"* | Historical sales charts, KPI metrics, trend analysis |
| *"What sales can I expect next week?"* | LightGBM 7-day forecast with confidence intervals |
| *"Why does the model predict this number?"* | SHAP waterfall — step-by-step explanation in euros |
| *"What if I run a promotion next week?"* | What-If simulator flips a promo toggle, re-runs forecast |
| *"Which of my 5 stores needs attention most?"* | AI agent ranks all stores by risk, evidence, and opportunity |
| *"Is my fleet healthy?"* | Fleet Health Score (0–100) combining accuracy + promo uplift |

---

## 👥 Team

| Person | Role | Owns |
|--------|------|------|
| **Parvat** *(Lead)* | Integration & Testing | `app.py`, `config.py`, `tests/test_curveball.py`, README |
| **Himanshu** | Data Engineer | `src/data_pipeline.py`, `src/database.py`, `notebooks/eda.ipynb` |
| **Ashutosh** | ML Engineer | `src/model_engine.py`, `src/feature_engineering.py` |
| **Saumya** | AI/Agent Engineer | `src/agent_graph.py`, `src/decision_engine.py`, `src/prompts.py`, guardrails |
| **Dikshit** | Frontend | All 5 Streamlit pages in `pages/` |

---

## ✨ Features

### 📊 Analytics Dashboard
- Fleet-wide KPI cards (total stores, avg daily sales, best/worst performer)
- 🩺 **Fleet Health Score** — animated ring gauge (0–100), computed from model accuracy + promo uplift
- 📉 **Worst Day of the Week** — auto-detected from real SQL query with action tip
- 🔔 **Sales Drop Alert Simulation** — configure store, threshold, and email; shows confirmation toast
- Multi-store sales trend, ranking bar charts, store type breakdown
- Anomaly detection timeline + promo history deep-dive

### 📈 Forecasting
- 7-day LightGBM forecast with confidence bands per store
- 🎯 **Forecast accuracy badge** — color-coded % accuracy in the KPI row
- 🎚️ **Animated confidence meter** — bar fills from 0 → computed confidence on page load
- SHAP driver bar + waterfall chart — explains forecast in real euros
- **What-If Simulator** — promo toggle re-runs forecast instantly
- 🏆 **Best day to run a promotion calendar** — 7-day heat-strip showing promo uplift per day

### 🔍 Promotion Analysis
- Fleet-wide promo uplift ranking
- Store-type breakdown of promo effectiveness
- Single-store promo deep-dive with timeline
- Downloadable promo activity table

### 🤖 AI Assistant
- Natural language → structured Observation / Prediction / Evidence / Recommendation
- 🎤 **Live voice input** — speak your question, it transcribes and submits automatically
- ⚡ **Response time badge** — shows answered-in time, persists in chat history
- Streaming response with live elapsed timer
- Citation cards showing every data source used
- Example question chips + "Surprise Me" style prompts
- Guardrails prevent hallucination — all numbers grounded in tool results

### ⚙️ Model Performance
- RMSPE, MAE, R² for LightGBM, XGBoost, and Baseline
- Walk-forward cross-validation diagram
- Model comparison chart and downloadable metrics table

---

## 🏗 Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit 1.35+ |
| **Primary ML Model** | LightGBM (RMSPE ≈ 0.12) |
| **Secondary ML Model** | XGBoost (comparison) |
| **Explainability** | SHAP |
| **AI Agent** | LangGraph + OpenRouter LLM |
| **Database** | SQLite (via SQLAlchemy) |
| **Visualisation** | Plotly |
| **Voice Input** | streamlit-mic-recorder + SpeechRecognition |
| **Testing** | pytest (288 tests, all passing) |
| **Container** | Docker + Docker Compose |

---

## 📁 Project Structure

```
capstone/
├── app.py                        # Streamlit entry point
├── config.py                     # All paths + feature flags
├── Dockerfile                    # One-command container build
├── docker-compose.yml            # docker compose up --build
├── CONTRIBUTING.md               # How to add new store types etc.
│
├── pages/
│   ├── 1_🏠_Dashboard.py         # Fleet analytics + health score + alerts
│   ├── 2_📈_Forecasting.py       # 7-day forecast + SHAP + promo calendar
│   ├── 3_🔍_Promotion_Analysis.py# Uplift ranking + deep-dive
│   ├── 4_🤖_AI_Assistant.py      # Streaming chat + voice input
│   └── 5_⚙️_Model_Performance.py # Metrics + CV diagram
│
├── src/
│   ├── data_pipeline.py          # ETL — cleans CSV, writes retail.db
│   ├── database.py               # All SQL query functions (single source of truth)
│   ├── feature_engineering.py    # Lag features, rolling stats, promo flags
│   ├── model_engine.py           # Forecast, SHAP, What-If, metrics functions
│   ├── agent_graph.py            # LangGraph AI agent (Router → Tools → Decision)
│   ├── decision_engine.py        # Obs/Prediction/Evidence/Recommendation builder
│   ├── prompts.py                # System prompts for the LLM
│   ├── guardrails.py             # Anti-hallucination + scope enforcement
│   ├── validation.py             # Response structure validation
│   └── query_understanding.py    # Intent classification + entity extraction
│
├── components/
│   ├── charts.py                 # Shared Plotly chart functions
│   └── ui_helpers.py             # kpi_card, section_header, citation_card, etc.
│
├── tests/                        # 288 tests across 9 test files
├── models/                       # Trained .pkl files (lgbm, xgboost, shap, medians)
├── data/                         # train.csv, store.csv → retail.db (not committed)
├── docs/                         # architecture.md, model_report.md, agent_design.md
└── notebooks/eda.ipynb           # Exploratory data analysis (9+ analyses)
```

---

## 🐳 Quick Start — Docker (One Command)

```bash
# Clone
git clone https://github.com/parvatkhattak/AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence.git
cd AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence

# Copy env and add your OpenRouter API key
cp .env.example .env
# edit .env → set OPENROUTER_API_KEY=sk-or-...

# Place your data files
# data/train.csv  (from Kaggle Rossmann dataset)
# data/store.csv

# Build and run — everything else is automatic
docker compose up --build
```

**App is live at `http://localhost:8501`**

The container will automatically:
1. Install all Python dependencies
2. Run `data_pipeline.py` to build `retail.db` if it doesn't exist
3. Start Streamlit on port 8501

> **Volume mount:** `./data` is mounted into the container, so `retail.db` survives rebuilds.

---

## 💻 Quick Start — Local

```bash
# 1. Clone & install
git clone https://github.com/parvatkhattak/AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence.git
cd AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Set OPENROUTER_API_KEY in .env

# 3. Place Rossmann data files in data/
#    data/train.csv  and  data/store.csv

# 4. Build the database
python src/data_pipeline.py

# 5. Train models (or use pre-trained .pkl files if committed)
python compare_models.py

# 6. Run
streamlit run app.py
```

---

## 📱 Application Pages

### 🏠 Dashboard
Fleet-wide command centre:
- **5 KPI cards** — Total stores, Avg daily sales, Total revenue, Top store, Bottom store
- **🩺 Fleet Health Score** — Ring gauge (0–100) computed from model accuracy + promo uplift
- **📉 Worst Day of the Week** — Actual SQL result: which day has the lowest fleet avg sales
- **🔔 Sales Drop Alert** — Set a store, a % threshold, and an email; simulates alert configuration
- Sales trend, store ranking charts, store type comparison, promo uplift ranking, anomaly detection

### 📈 Forecasting
Per-store prediction engine:
- **Store selector** → triggers 7-day LightGBM forecast with confidence bands
- **5 KPI cards** including 🎯 Model Accuracy badge
- **Animated confidence meter** — bar animates to show how tight the forecast interval is
- **SHAP bar + waterfall** — explains forecast in plain-English euro terms
- **What-If Simulator** — promo toggle shows impact vs no-promo scenario
- **🏆 Best promotion day calendar** — 7-day heat strip, trophy on best uplift day

### 🔍 Promotion Analysis
- Fleet-wide uplift ranking chart
- Store-type breakdown (A vs B vs C vs D effectiveness)
- Single-store promo vs non-promo sales comparison
- Downloadable ranked summary table

### 🤖 AI Assistant
- Type **or speak** (🎤 voice input) any question about your stores
- Real-time streaming answer with live stage indicator and elapsed timer
- **⚡ Response time badge** — color-coded pill (green <5s, amber <15s, red) persists in history
- Structured answer: **Observation → Prediction → Evidence → Recommendation**
- Citation cards showing exactly which database functions and model calls were used
- Guardrails prevent the LLM from inventing numbers not in tool results
- Example question chips and "Clear Chat" button

### ⚙️ Model Performance
- RMSPE / MAE / R² cards for LightGBM, XGBoost, and Baseline
- Walk-forward cross-validation diagram
- Model comparison grouped bar chart
- Downloadable metrics table

---

## 🤖 How the AI Works

```
User types or speaks a question
           │
    app.py (Streamlit)
           │
    query_understanding.py ── intent classification + entity extraction
           │
    guardrails.py ──────────── scope check (retail only)
           │
    agent_graph.py (LangGraph)
           │
    ┌──────┴──────┐
    │             │
database.py   model_engine.py
(SQLite)      (LightGBM + SHAP)
    │             │
    └──────┬──────┘
           │
    decision_engine.py ─── builds Obs/Pred/Evidence/Rec
           │
    response_validation.py ─ checks output is grounded
           │
    Streamed back to the user with citation cards
```

### 🔒 Hallucination Prevention

Every number in a response must come from an actual tool result. The system enforces:
- All figures traced back to `database.py` or `model_engine.py` outputs
- `guardrails.py` blocks out-of-scope questions before the LLM runs
- `response_validation.py` verifies the answer structure before display
- Every response includes a **Sources:** section listing the exact functions called

---

## 📦 Dataset

**Source:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data)

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | ~1,017,209 | Daily sales for 1,115 stores (Jan 2013 – Jul 2015) |
| `store.csv` | 1,115 | Store attributes: type, competition, promotions |

### Key Fields

| Field | Source | Used For |
|-------|--------|---------|
| `Sales` | train.csv | Forecast target variable |
| `Promo` | train.csv | Promotional analysis + What-If |
| `StoreType` | store.csv (joined) | Store segmentation, type breakdown |
| `CompetitionDistance` | store.csv | Competition impact in SHAP |
| `Promo2` / `IsPromo2Active` | store.csv | Continuous promo tracking |
| `DayOfWeek` | train.csv | Worst-day-of-week analysis |

---

## 🧪 Running Tests

```bash
# Run all 288 tests
pytest tests/ -q

# Run a specific file
pytest tests/test_model_engine.py -v
pytest tests/test_database.py -v
pytest tests/test_curveball.py -v -s    # presentation stress test

# Coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Files

| File | Covers |
|------|--------|
| `test_data_pipeline.py` | ETL, no nulls, no data leakage in lags |
| `test_database.py` | All 8 DB functions — column names, types, values |
| `test_model_engine.py` | Forecast (7 rows), SHAP structure, What-If, metrics |
| `test_agent_graph.py` | Agent doesn't crash, response has 4 sections |
| `test_decision_engine.py` | Decision report has all fields, citations non-empty |
| `test_guardrails.py` | Out-of-scope queries blocked, in-scope allowed |
| `test_query_understanding.py` | Intent classification, entity extraction, follow-ups |
| `test_agent_behaviour.py` | End-to-end agent behaviour scenarios |
| `test_curveball.py` | 20× stress test of the live presentation question |

---

## 🎯 The Curveball Question

> *"I manage Stores 100, 200, 300, 400, and 500. Based on historical performance and your forecast, which stores should I focus on next week, why, and what does their promotional history tell me?"*

The agent automatically:
1. Extracts `[100, 200, 300, 400, 500]` from the question
2. Fetches 30-day historical sales for all 5 stores
3. Runs the 7-day LightGBM forecast for each
4. Fetches promotional uplift history for each
5. Ranks by composite risk score (declining trend + weak forecast + unused promo potential)
6. Returns a structured response:

```
🔍 OBSERVATION
Store 200 has seen an 18% sales decline over the past 4 weeks.

📈 PREDICTION
Our model forecasts Store 200 at €9,200/day next week — 14% below fleet average.

📊 EVIDENCE
SHAP shows promotional activity contributes +31% uplift for Store 200 on promo days.
Store 200 has had 0 promo days in the past 3 weeks.

✅ RECOMMENDATION
Activate Promo 1 for Store 200 next week. Historical data strongly supports this.

📚 Sources: database.get_store_metrics, model_engine.get_7day_forecast,
            database.get_promo_history
```

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| **Dataset ends July 2015** | Forecasts are simulated, not live |
| **SQLite concurrency** | Single-user only — not for concurrent production use |
| **7-day horizon** | Model unreliable beyond 7 days without additional long-horizon features |
| **LLM latency** | AI responses take 5–20s depending on OpenRouter load |
| **Voice input** | Requires browser microphone permission; works in Chrome/Edge |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/architecture.md`](docs/architecture.md) | Full technical design, module graph, data model, API contracts |
| [`docs/model_report.md`](docs/model_report.md) | Model selection, walk-forward CV, SHAP guide |
| [`docs/agent_design.md`](docs/agent_design.md) | LangGraph graph, guardrails, prompt engineering |
| [`docs/data_assumptions.md`](docs/data_assumptions.md) | Every data cleaning decision |
| [`docs/architecture_diagram.jpg`](docs/architecture_diagram.jpg) | Visual system architecture diagram |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to add new store types, branching, PR checklist |
| [`TEAM_PLAN.md`](TEAM_PLAN.md) | Sprint plan, roles, daily task breakdown |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including a detailed **8-step walkthrough for adding a new store type**.

Quick summary:
```bash
git checkout -b feature/<your-feature>
# make changes
pytest tests/ -q          # must pass
git push origin feature/<your-feature>
# open a PR → tag Parvat for review
```

---

<div align="center">
  <strong>Built by Team Retail AI · Capstone 2026</strong><br/>
  <em>Analyze · Predict · Explain · Recommend</em>
</div>

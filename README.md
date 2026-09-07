# 🛒 Retail AI — Sales Forecasting & Decision Intelligence

> **An AI-powered platform that helps retail store managers analyze performance, predict future sales, and receive evidence-backed recommendations — all through a natural language interface.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-green)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Live Demo Screenshot](#-live-demo-screenshot)
- [Team](#-team)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [How to Run](#-how-to-run)
- [How the AI Works](#-how-the-ai-works)
- [Application Pages](#-application-pages)
- [Dataset](#-dataset)
- [Running Tests](#-running-tests)
- [The Curveball Question](#-the-curveball-question)
- [Known Limitations](#-known-limitations)
- [Documentation](#-documentation)

---

## 🎯 What This Project Does

This platform answers real business questions for a retail manager who operates 1,000+ stores:

| Question | How We Answer It |
|----------|-----------------|
| *"How is Store 125 performing?"* | Historical sales charts, KPI metrics, trend analysis |
| *"What sales can I expect next week?"* | XGBoost 7-day forecast with confidence intervals |
| *"Which stores are likely to underperform?"* | Risk scoring based on trend + forecast |
| *"How have promotions performed for Store 35?"* | Promo uplift analysis, SHAP feature contribution |
| *"Which of my 5 stores should I focus on?"* | Multi-store Decision Intelligence report |

Every answer is **grounded in real data** — the AI never fabricates numbers.

The system follows the **ANALYZE → PREDICT → EXPLAIN → RECOMMEND** framework:

```
📊 ANALYZE        📈 PREDICT        💡 EXPLAIN        ✅ RECOMMEND
Historical EDA  →  7-day XGBoost  →  SHAP drivers  →  Action + Evidence
```

---

## 👥 Team

| Name | Role | Responsible For |
|------|------|----------------|
| **Dev 1** | Team Lead & Integration | `app.py`, `config.py`, system integration, tests |
| **Dev 2** | Data Engineer & EDA | `data_pipeline.py`, `database.py`, EDA notebook |
| **Dev 3** | ML & Explainability | `model_engine.py`, XGBoost, SHAP |
| **Dev 4** | Agentic AI & Decision | `agent_graph.py`, `decision_engine.py`, LLM prompts |
| **Dev 5** | Frontend & UI | All Streamlit pages, charts, components |

---

## ✨ Features

### 📊 Performance Dashboard
- Fleet-wide KPI cards (total sales, average daily, top/bottom store)
- Sales trend charts across time periods
- Store type (A/B/C/D) performance comparison
- Holiday impact analysis
- Anomaly detection — flags stores with unusual sales behaviour
- Store clustering — groups stores by behavioural similarity

### 📈 Sales Forecasting
- **7-day forecast** for any of the 1,115 stores
- **Confidence interval bands** (10th–90th percentile)
- Comparison: Moving Average baseline vs XGBoost vs LightGBM
- **SHAP waterfall chart** — explains exactly why each prediction was made
- **What-If Simulator** — toggle a promotion on/off and see how sales change

### 🔍 Promotion Analysis
- Promo vs non-promo sales uplift per store
- Fleet-wide promo uplift ranking
- Store type breakdown of promotional effectiveness
- Promo activity timeline

### 🤖 AI Assistant
- Natural language questions in plain English
- Structured responses: **Observation → Prediction → Evidence → Recommendation**
- **Streaming responses** (real-time, like ChatGPT)
- **Citation cards** — every fact shows which data source it came from
- Session conversation history
- Example question chips for quick exploration

### ⚙️ Model Performance
- RMSPE, MAE, R² metrics
- Walk-forward cross-validation results
- Baseline vs ML model comparison
- Global SHAP feature importance (beeswarm)

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.11 | Core language |
| **Data** | Pandas, NumPy | Data manipulation |
| **Database** | SQLite + SQLAlchemy | Local analytical store |
| **ML** | XGBoost, LightGBM, scikit-learn | Forecasting models |
| **Explainability** | SHAP | Feature attribution |
| **AI Agent** | LangGraph | Agent orchestration |
| **LLM** | OpenRouter (Llama 3 / Claude) | Natural language responses |
| **App** | Streamlit | Web interface |
| **Charts** | Plotly | Interactive visualisations |
| **Testing** | pytest | Automated tests |

---

## 📁 Project Structure

```
capstone/
│
├── app.py                        # Main entry point — run this to start the app
├── config.py                     # All shared constants (paths, model names, LLM config)
├── requirements.txt              # All Python dependencies
├── setup.sh                      # One-command environment setup
├── .env.example                  # Template for your API key
│
├── data/
│   ├── train.csv                 # Raw Rossmann training data (1M+ rows)
│   ├── store.csv                 # Store characteristics (1115 stores)
│   ├── retail.db                 # Generated SQLite database (created by setup)
│   └── mocks/                    # Mock data files (for development)
│
├── src/
│   ├── data_pipeline.py          # Cleans data, engineers features, populates SQLite
│   ├── database.py               # All database query functions
│   ├── feature_engineering.py    # Lag, rolling, calendar features
│   ├── model_engine.py           # Train, forecast, SHAP, What-If
│   ├── agent_graph.py            # LangGraph agent definition
│   ├── decision_engine.py        # Observation → Recommendation logic
│   ├── prompts.py                # LLM system prompts
│   └── utils.py                  # Shared utility functions
│
├── pages/
│   ├── 1_🏠_Dashboard.py
│   ├── 2_📈_Forecasting.py
│   ├── 3_🔍_Promotion_Analysis.py
│   ├── 4_🤖_AI_Assistant.py
│   └── 5_⚙️_Model_Performance.py
│
├── components/
│   ├── charts.py                 # Reusable Plotly chart functions
│   ├── ui_helpers.py             # Reusable Streamlit widgets
│   └── report_generator.py       # CSV/PDF export
│
├── models/
│   ├── xgboost_model.pkl         # Trained XGBoost model (generated by setup)
│   ├── lgbm_model.pkl            # Trained LightGBM model
│   └── shap_explainer.pkl        # SHAP explainer object
│
├── tests/
│   ├── conftest.py               # Shared test fixtures
│   ├── test_data_pipeline.py
│   ├── test_database.py
│   ├── test_model_engine.py
│   ├── test_agent_graph.py
│   ├── test_decision_engine.py
│   └── test_curveball.py         # Presentation stress test (20 runs)
│
├── notebooks/
│   ├── eda.ipynb                 # Full exploratory data analysis
│   ├── feature_analysis.ipynb    # Feature engineering rationale
│   └── model_experiments.ipynb   # Model comparison and selection
│
└── docs/
    ├── architecture.md           # Full system architecture (this project's design doc)
    ├── data_assumptions.md       # Data cleaning decisions
    ├── model_report.md           # ML model selection and metrics
    └── agent_design.md           # Agent architecture and prompt engineering
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- The Rossmann dataset files: `train.csv` and `store.csv` in the `data/` folder
- An [OpenRouter](https://openrouter.ai) API key (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/your-team/retail-ai-capstone.git
cd retail-ai-capstone
```

### 2. Set Up Environment (One Command)

```bash
bash setup.sh
```

This script will:
- Create a Python virtual environment
- Install all dependencies from `requirements.txt`
- Copy `.env.example` to `.env`

### 3. Add Your API Key

Open the `.env` file and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_key_here
```

> **No key?** Get a free one at [openrouter.ai](https://openrouter.ai). The free tier includes access to Llama 3.

### 4. Build the Database and Train the Model

```bash
# Step 1: Clean data and populate SQLite database
python src/data_pipeline.py

# Step 2: Train the forecasting model and compute SHAP values
python src/model_engine.py --train
```

> ⏱️ **Expected times:** Database build ~3–5 min | Model training ~5–10 min

### 5. Launch the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 🖥️ How to Run

### Run the Full Pipeline (fresh install)

```bash
# Activate virtual environment
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Build everything
python src/data_pipeline.py
python src/model_engine.py --train

# Start the app
streamlit run app.py
```

### Just Start the App (database + model already built)

```bash
source venv/bin/activate
streamlit run app.py
```

### Run the EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 🧠 How the AI Works

The AI Assistant is built on **LangGraph**, a graph-based agent framework. When you type a question, this is what happens:

```
Your question
     │
     ▼
 Router Node ──► Classifies intent (performance / forecast / recommend / what-if)
     │
     ├──► Data Analyst Node ──► Queries SQLite database (real historical sales)
     │
     ├──► Forecast Node ──► Runs XGBoost model (real 7-day predictions)
     │
     ├──► Decision Node ──► Combines both → Observation → Prediction → Evidence → Recommendation
     │
     └──► What-If Node ──► Simulates forecast with/without promotion
              │
              ▼
         LLM receives real data, formats a clear response
              │
              ▼
         Response streams back to you with source citations
```

### 🔒 Hallucination Prevention

The AI is strictly instructed:
> *"You may only state numbers that appear in the tool results provided. If a figure is not in the tool results, say 'data not available'. Never estimate or guess."*

Every response includes a **[Sources]** section showing which database queries and model functions provided each piece of data.

---

## 📱 Application Pages

### 🏠 Dashboard
Your first stop. Shows a bird's-eye view of all 1,115 stores:
- Total fleet sales, average daily sales, best and worst performing stores
- Sales trend over time with period selector
- Store type performance breakdown (A, B, C, D)
- Anomaly flags — which stores had unusual sales recently?
- Store clustering map — which stores behave similarly?

### 📈 Forecasting
Pick any store and instantly see:
- 7-day sales forecast with upper/lower confidence bounds
- Side-by-side comparison: your model vs simple moving average baseline
- SHAP waterfall chart: *"why does the model predict this number?"*
- **What-If Simulator:** flip a promo toggle and see the forecast change in real time

### 🔍 Promotion Analysis
Deep dive into promotional effectiveness:
- Which stores have the highest promotional uplift?
- How does promo effectiveness differ between store types?
- Visual comparison of promo vs non-promo periods for any store
- Timeline of promotional activity

### 🤖 AI Assistant
Ask anything in plain English. Examples:
- *"How is Store 125 performing?"*
- *"Compare Store 125 and Store 220"*
- *"What are expected sales for Store 100 next week?"*
- *"Which stores are likely to underperform next week?"*
- *"How have promotions performed for Store 35?"*
- *"I manage Stores 100, 200, 300, 400, 500 — which should I focus on?"*

Responses always include: **Observation → Prediction → Evidence → Recommendation**

### ⚙️ Model Performance
Full transparency on how the forecasting model was built:
- RMSPE, MAE, R² scores
- Walk-forward cross-validation results across 5 folds
- XGBoost vs LightGBM vs Moving Average comparison
- Global feature importance (which features matter most across all stores?)

---

## 📦 Dataset

**Source:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data)

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | ~1,017,209 | Daily sales for 1,115 stores over 3 years |
| `store.csv` | 1,115 | Store attributes: type, competition, promotions |

### Key Fields Used

| Field | Source | Used For |
|-------|--------|---------|
| `Sales` | train.csv | Target variable for forecasting |
| `Promo` | train.csv | Promotional analysis |
| `StateHoliday` | train.csv | Holiday impact analysis |
| `StoreType` | store.csv | Store segmentation |
| `CompetitionDistance` | store.csv | Competition impact analysis |
| `Promo2` | store.csv | Continuous promotion tracking |

> The dataset covers **January 2013 – July 2015**. Forecasts are relative to the dataset's end date.

---

## 🧪 Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run a Specific Test File

```bash
pytest tests/test_database.py -v
pytest tests/test_model_engine.py -v
pytest tests/test_agent_graph.py -v
```

### Run the Presentation Stress Test

```bash
pytest tests/test_curveball.py -v -s
```

This runs the live curveball question 20 times and verifies:
- ✅ Zero crashes
- ✅ Consistent store ranking (#1 ranked store appears in ≥18/20 runs)
- ✅ Response time < 30 seconds per query
- ✅ No hallucinated numbers in any response

### Check Test Coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🎯 The Curveball Question

> *"I manage Stores 100, 200, 300, and 500. Based on historical performance and your forecast, which stores should I focus on next week, why, and what does their promotional history tell me?"*

This is the live demonstration question. The system handles it by:

1. Extracting store IDs `[100, 200, 300, 400, 500]` from the query
2. Fetching 30 days of historical sales for all 5 stores
3. Running the 7-day XGBoost forecast for each store
4. Fetching promotional uplift history for each store
5. Ranking stores by composite risk score (trend decline + forecast weakness + unused promo potential)
6. Returning a structured decision report:

```
🔍 OBSERVATION
Store 200 has seen an 18% sales decline over the past 4 weeks,
the steepest fall in this group.

📈 PREDICTION
The model forecasts Store 200 will average €9,200/day next week
— 14% below the group average of €10,700/day.

📊 EVIDENCE
SHAP analysis shows promotional activity contributes +31% to
Store 200's sales on promo days. Store 200 has had 0 promo days
in the past 3 weeks.

✅ RECOMMENDATION
Priority: Store 200. Activating Promo 1 next week is strongly
supported by historical data. All other stores show stable or
improving trends.

📚 Sources: database.get_store_metrics, model_engine.get_7day_forecast,
            database.get_promo_history
```

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| **Dataset period** | Data ends July 2015 — forecasts are simulated, not live |
| **Local only** | Designed to run on a laptop; not production-scaled |
| **7-day horizon** | Model is not reliable beyond 7-day forecasts |
| **LLM speed** | AI responses may take 10–20 seconds depending on the LLM |
| **SQLite concurrency** | Single-user only — SQLite doesn't support concurrent writes |

---

## 📚 Documentation

| Document | Location | Description |
|----------|----------|-------------|
| System Architecture | [`docs/architecture.md`](docs/architecture.md) | Full technical design, diagrams, data model |
| Team Work Plan | [`TEAM_PLAN.md`](TEAM_PLAN.md) | Sprint plan, roles, daily tasks |
| Data Assumptions | [`docs/data_assumptions.md`](docs/data_assumptions.md) | Every data cleaning decision |
| Model Report | [`docs/model_report.md`](docs/model_report.md) | ML model selection, metrics, SHAP guide |
| Agent Design | [`docs/agent_design.md`](docs/agent_design.md) | LangGraph architecture, prompt engineering |

---

## 🤝 Contributing (Team Members)

### Branch Naming

```bash
git checkout -b feature/data-pipeline    # Dev 2
git checkout -b feature/ml-engine        # Dev 3
git checkout -b feature/agent-router     # Dev 4
git checkout -b feature/streamlit-ui     # Dev 5
```

### Before Raising a PR

```bash
# Run your tests
pytest tests/test_<your_module>.py -v

# Check for import errors
python -c "from src.<your_module> import *; print('OK')"
```

### Never Change a Shared Function Signature Without Asking Dev 1 First

The following functions are called by multiple team members. Changing them without coordination will break everyone's code:

- `database.get_store_metrics()`
- `database.get_promo_history()`
- `model_engine.get_7day_forecast()`
- `model_engine.get_shap_explanations()`
- `agent_graph.run_agent_stream()`

---

## 📝 License

This project was built as an academic capstone. Dataset sourced from [Rossmann Store Sales on Kaggle](https://www.kaggle.com/c/rossmann-store-sales).

---

<div align="center">
  <strong>Built by Team Retail AI · Capstone 2025</strong><br/>
  <em>Analyze · Predict · Explain · Recommend</em>
</div>

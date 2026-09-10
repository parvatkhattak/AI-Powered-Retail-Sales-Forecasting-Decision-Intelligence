# 🎤 DAY 1 CAPSTONE PRESENTATION GUIDE & FULL CODEBASE AUDIT

**Project Title:** AI-Powered Retail Sales Forecasting & Decision Intelligence Platform  
**Presenter:** Parvat Khattak (Tech Lead & Integrator)  
**Team Members:** Himanshu, Ashutosh, Saumya, Dikshit  
**Date:** Day 1 Completion & Sync  

---

## 📋 EXECUTIVE SUMMARY (Use this for your 1-minute intro)

> "Good morning/afternoon everyone! Today we present Day 1 of our AI-Powered Retail Sales Forecasting & Decision Intelligence platform built on the Rossmann Store Sales dataset (1,115 stores, 1M+ sales records).
> 
> Our mission is to bridge the gap between complex machine learning forecasts and real-world retail decision-making. We don't just predict sales numbers—we tell store managers **why** sales are changing and **what concrete inventory or promotional actions** they should take using an integrated LLM AI Agent powered by `meta-llama/llama-3.3-70b-instruct:free` via OpenRouter."

---

## 👥 SECTION 1: TEAM WORK BREAKDOWN & DETAILED RATIONALE ("WHY")

When presenting team contributions, use this section to explain **what** each member built and **why** it was necessary:

### 1. 👨‍💻 Himanshu — Data Engineering & SQLite Database Architecture
* **What He Did:**
  1. Built the clean data pipeline in [`src/data_pipeline.py`](file:///home/parvat-khattak/Downloads/capstone/src/data_pipeline.py).
  2. Implemented the SQLite database schema in [`src/database.py`](file:///home/parvat-khattak/Downloads/capstone/src/database.py).
  3. Created store-level KPI metric aggregation utilities and mock data fallbacks in [`data/mocks/mock_store_metrics.json`](file:///home/parvat-khattak/Downloads/capstone/data/mocks/mock_store_metrics.json).
* **Why He Did It (Rationale):**
  * Raw retail data contains missing entries (e.g. missing `CompetitionDistance`, `Promo2SinceYear`, closed store zero-sales days). Cleaning ensures mathematical models don't encounter errors during training.
  * Storing processed data in SQLite (`data/retail.db`) allows fast SQL indexing and query execution for our UI and AI Agent, avoiding slow 100MB+ CSV re-reads into memory on every click.
  * Mock files decoupled frontend development from backend pipeline generation.

---

### 2. 👨‍💻 Ashutosh — Feature Engineering & Baseline Modeling
* **What He Did:**
  1. Built lag and rolling statistical features in [`src/feature_engineering.py`](file:///home/parvat-khattak/Downloads/capstone/src/feature_engineering.py).
  2. Implemented a Simple Moving Average baseline forecasting model.
  3. Generated [`data/mocks/mock_forecast.json`](file:///home/parvat-khattak/Downloads/capstone/data/mocks/mock_forecast.json) and [`data/mocks/mock_shap.json`](file:///home/parvat-khattak/Downloads/capstone/data/mocks/mock_shap.json).
* **Why He Did It (Rationale):**
  * Time-series data has strong autocorrelation (today's sales depend on last week's sales). Creating lag features (`Sales_Lag_7`, `Sales_Lag_14`, 7-day rolling mean/std) transforms raw dates into tabular features compatible with tree-based machine learning models (XGBoost, LightGBM).
  * Building a simple baseline model establishes an empirical benchmark (RMSPE ~0.3459) to demonstrate exactly how much accuracy our advanced gradient boosting models gain over naive estimates.

---

### 3. 👩‍💻 Saumya — Agentic AI Architecture & LLM Integration
* **What She Did:**
  1. Constructed the multi-node agent workflow in [`src/agent_graph.py`](file:///home/parvat-khattak/Downloads/capstone/src/agent_graph.py) using `langgraph`.
  2. Integrated OpenRouter API support for `meta-llama/llama-3.3-70b-instruct:free`.
  3. Designed structured prompt templates and intent routers in [`src/prompts.py`](file:///home/parvat-khattak/Downloads/capstone/src/prompts.py).
* **Why She Did It (Rationale):**
  * Retail managers do not want raw arrays of float numbers or complex statistical plots. They need an intelligent assistant that can answer natural questions like *"What is the 7-day forecast for Store 1 and should I run a promo?"*
  * Using LangGraph state graphs ensures deterministic, safe execution (Query Classification -> Tool Call Execution -> Natural Language Formatting) so the LLM cannot invent non-existent numbers.

---

### 4. 👨‍💻 Dikshit — Multi-Page Streamlit UI & Chart Shell
* **What He Did:**
  1. Built the Streamlit app shell ([`app.py`](file:///home/parvat-khattak/Downloads/capstone/app.py)) and multi-page routing ([`pages/*.py`](file:///home/parvat-khattak/Downloads/capstone/pages)).
  2. Created custom responsive Plotly chart renderers in [`components/charts.py`](file:///home/parvat-khattak/Downloads/capstone/components/charts.py) and KPI card helpers in [`components/ui_helpers.py`](file:///home/parvat-khattak/Downloads/capstone/components/ui_helpers.py).
* **Why He Did It (Rationale):**
  * Provides a modern, dark-mode responsive dashboard split across executive views (Dashboard, Forecasting, Promotion Analysis, AI Assistant, Model Performance) rather than cluttering a single long web page.

---

### 5. 👨‍💼 Parvat (Tech Lead & Integrator) — Model Suite, Decision Rules & Integration
* **What I Did:**
  1. Developed [`src/model_engine.py`](file:///home/parvat-khattak/Downloads/capstone/src/model_engine.py) covering 5 distinct models: Baseline, Linear Regression, Random Forest, XGBoost, and LightGBM.
  2. Created model evaluation and tuning pipelines ([`compare_models.py`](file:///home/parvat-khattak/Downloads/capstone/compare_models.py), [`tune_models.py`](file:///home/parvat-khattak/Downloads/capstone/tune_models.py), [`shap_analysis.py`](file:///home/parvat-khattak/Downloads/capstone/shap_analysis.py)).
  3. Created the business rule decision engine in [`src/decision_engine.py`](file:///home/parvat-khattak/Downloads/capstone/src/decision_engine.py).
  4. Verified module imports, managed clean code integration, and updated project configuration.
* **Why I Did It (Rationale):**
  * Achieved state-of-the-art model performance (LightGBM RMSPE: **0.1177**, $R^2$: **0.9209**) evaluated on the official competition metric (Root Mean Square Percentage Error).
  * Automated decision logic maps ML outputs into actionable business recommendations (reorder stock alerts, promo uplift multipliers).

---

## 📂 SECTION 2: FILE-BY-FILE CODEBASE BLUEPRINT

This reference table outlines **which file does what**, **what it generates on execution**, and **where outputs are saved**:

| File Path | Description & Purpose | Output Generated on Execution | Output Storage Location |
| :--- | :--- | :--- | :--- |
| [`config.py`](file:///home/parvat-khattak/Downloads/capstone/config.py) | Central project configuration (paths, DB settings, hyperparameters, LLM settings). | Python configuration constants loaded by all scripts. | In-memory configuration |
| [`src/data_pipeline.py`](file:///home/parvat-khattak/Downloads/capstone/src/data_pipeline.py) | Reads raw `train.csv` & `store.csv`, cleans missing values, merges datasets. | Clean merged DataFrame & table entries. | Passed to memory / `data/retail.db` |
| [`src/database.py`](file:///home/parvat-khattak/Downloads/capstone/src/database.py) | SQLite database manager & table creation script. | SQLite relational database file. | [`data/retail.db`](file:///home/parvat-khattak/Downloads/capstone/data/retail.db) |
| [`src/feature_engineering.py`](file:///home/parvat-khattak/Downloads/capstone/src/feature_engineering.py) | Computes 7d/14d/28d lag columns, rolling means, and calendar features. | Engineered feature matrix (20+ columns). | In-memory DataFrame |
| [`src/model_engine.py`](file:///home/parvat-khattak/Downloads/capstone/src/model_engine.py) | Unified engine for training Baseline, Linear Regression, Random Forest, XGBoost, and LightGBM models. | Model objects & 7-day forecast dataframes. | In-memory models & serialized PKL files |
| [`compare_models.py`](file:///home/parvat-khattak/Downloads/capstone/compare_models.py) | Trains & benchmarks all 5 models against validation set using RMSPE, MAE, R². | Comparison CSV table, scatter plot chart, trained XGBoost & LightGBM PKL models. | [`model_comparison_results.csv`](file:///home/parvat-khattak/Downloads/capstone/model_comparison_results.csv), [`model_comparison_scatter.png`](file:///home/parvat-khattak/Downloads/capstone/model_comparison_scatter.png), [`models/lgbm_model.pkl`](file:///home/parvat-khattak/Downloads/capstone/models/lgbm_model.pkl), [`models/xgboost_model.pkl`](file:///home/parvat-khattak/Downloads/capstone/models/xgboost_model.pkl) |
| [`tune_models.py`](file:///home/parvat-khattak/Downloads/capstone/tune_models.py) | Optuna hyperparameter optimization script for LightGBM and XGBoost. | Best hyperparameter dictionary & trial logs printed to console. | Console logs / Model parameters |
| [`shap_analysis.py`](file:///home/parvat-khattak/Downloads/capstone/shap_analysis.py) | TreeSHAP feature importance analysis script. | SHAP Explainer model binary & summary bar plot PNG. | [`models/shap_explainer.pkl`](file:///home/parvat-khattak/Downloads/capstone/models/shap_explainer.pkl), [`shap_feature_importance.png`](file:///home/parvat-khattak/Downloads/capstone/shap_feature_importance.png) |
| [`src/decision_engine.py`](file:///home/parvat-khattak/Downloads/capstone/src/decision_engine.py) | Business rule engine for inventory safety stock calculations & promo strategy. | Recommendation dicts (e.g. stock reorder alerts, promo uplift %). | In-memory JSON dicts |
| [`src/agent_graph.py`](file:///home/parvat-khattak/Downloads/capstone/src/agent_graph.py) | LangGraph agent state machine for intent routing and LLM execution. | Agent responses using OpenRouter `meta-llama/llama-3.3-70b-instruct:free`. | Natural language responses |
| [`src/prompts.py`](file:///home/parvat-khattak/Downloads/capstone/src/prompts.py) | System prompt templates & intent classification guidelines. | Formatted string prompts for LLM. | In-memory string variables |
| [`components/charts.py`](file:///home/parvat-khattak/Downloads/capstone/components/charts.py) | Reusable Plotly charting functions (sales trends, forecasts, SHAP, promo uplift). | Interactive Plotly figure objects. | Rendered in Streamlit UI |
| [`components/ui_helpers.py`](file:///home/parvat-khattak/Downloads/capstone/components/ui_helpers.py) | Streamlit HTML/CSS metric card renderers and page styling helpers. | HTML styled KPI containers. | Rendered in Streamlit UI |
| [`app.py`](file:///home/parvat-khattak/Downloads/capstone/app.py) | Main application entry point for Streamlit. | Multi-page sidebar web application. | Local web server (`http://localhost:8501`) |
| [`pages/1_🏠_Dashboard.py`](file:///home/parvat-khattak/Downloads/capstone/pages/1_🏠_Dashboard.py) | Page 1: Executive KPI overview & sales trends. | Interactive charts & metric summaries. | Web UI Page |
| [`pages/2_📈_Forecasting.py`](file:///home/parvat-khattak/Downloads/capstone/pages/2_📈_Forecasting.py) | Page 2: Store forecast selector, confidence bands & SHAP importance. | 7-day forecast plots & feature importance. | Web UI Page |
| [`pages/3_🔍_Promotion_Analysis.py`](file:///home/parvat-khattak/Downloads/capstone/pages/3_🔍_Promotion_Analysis.py) | Page 3: Promotional impact & sales uplift analysis. | Promo uplift charts & store comparison. | Web UI Page |
| [`pages/4_🤖_AI_Assistant.py`](file:///home/parvat-khattak/Downloads/capstone/pages/4_🤖_AI_Assistant.py) | Page 4: Interactive chat UI powered by LangGraph AI Agent. | Conversational AI chat with source citations. | Web UI Page |
| [`pages/5_⚙️_Model_Performance.py`](file:///home/parvat-khattak/Downloads/capstone/pages/5_⚙️_Model_Performance.py) | Page 5: Model accuracy leaderboard & residual diagnostics. | Metric tables (RMSPE, MAE, R²) & scatter plots. | Web UI Page |
| [`data/mocks/generate_mocks.py`](file:///home/parvat-khattak/Downloads/capstone/data/mocks/generate_mocks.py) | Generates fallback mock JSON files for offline testing. | Mock data JSON files. | [`data/mocks/*.json`](file:///home/parvat-khattak/Downloads/capstone/data/mocks/) |

---

## 🏆 SECTION 3: MODEL PERFORMANCE RESULTS (Key Presentation Slide)

When presenting model performance, highlight these exact numbers achieved on the validation dataset:

| Model Name | Train RMSPE | Valid RMSPE | Valid MAE | Valid $R^2$ | Execution Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Moving Average** *(Baseline)* | N/A | `0.3459` | 1,420 | `0.4120` | <0.1s | Benchmark |
| **Linear Regression** | `0.3034` | `0.2223` | 950 | `0.7184` | 0.7s | Linear Baseline |
| **Random Forest** | `0.1412` | `0.1582` | 680 | `0.8540` | 12.4s | Ensemble |
| **XGBoost** | `0.1150` | `0.1245` | 512 | `0.9080` | 4.2s | High Accuracy |
| 🥇 **LightGBM** *(Best Model)* | `0.1080` | **`0.1177`** | **478** | **`0.9209`** | **1.8s** | **Selected Winner** |

> **Key takeaway to read aloud:** *"Our LightGBM model reduced forecast error (RMSPE) from 34.6% down to 11.7% while explaining over 92% of sales variance ($R^2 = 0.9209$)."*

---

## 🏃 SECTION 4: HOW TO RUN & DEMO THE APPLICATION (Live Presentation Steps)

### Step 1: Environment Setup
```bash
# Clone project directory and activate environment
cd capstone
source venv/bin/activate

# Install requirements (cleaned: removed unused packages)
pip install -r requirements.txt
```

### Step 2: Initialize Database & Run Machine Learning Pipeline
```bash
# 1. Clean raw data & populate SQLite database
python3 src/data_pipeline.py

# 2. Train & compare all models (generates model comparison CSV & PNG)
python3 compare_models.py

# 3. Generate SHAP feature importance plot
python3 shap_analysis.py
```

### Step 3: Launch Streamlit App
```bash
streamlit run app.py
```
*Open browser at `http://localhost:8501` to show all 5 multi-page dashboards.*

### Step 4: Demo the AI Assistant (Page 4)
Ask the assistant live sample questions:
* *"What is the forecasted sales for store 1 next week?"*
* *"Should Store 5 run a promotion next week?"*
* *"Which features are driving store sales the most?"*

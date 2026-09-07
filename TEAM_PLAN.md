# 🛒 Retail AI Capstone — Team Work Plan
### Who Does What, On Which Day

> **Project:** AI-Powered Retail Sales Forecasting & Decision Intelligence
> **Team:** 5 Developers | **Duration:** 4 Days | **Goal:** S-Grade

---

## 👥 Meet the Team

| Person | Role | What They Own |
|--------|------|---------------|
| **Parvat** *(Team Lead)* | Lead + Integration | Ties everything together; sets up the project; runs final tests |
| **Himanshu** | Data Engineer & Analyst | Cleans the data; builds the database; does all exploratory charts |
| **Ashutosh** | Machine Learning Engineer | Trains the forecasting model; explains predictions using SHAP |
| **Saumya** | AI & Agent Engineer | Builds the AI assistant; creates the decision recommendation engine |
| **Dikshit** | Frontend Developer | Builds every screen the user sees in the app |

---

## 🗂️ What Each Person Builds

| Person | Files They Write |
|--------|-----------------|
| Parvat | `app.py`, `config.py`, `README.md`, `tests/test_curveball.py` |
| Himanshu | `src/data_pipeline.py`, `src/database.py`, `notebooks/eda.ipynb` |
| Ashutosh | `src/model_engine.py`, `src/feature_engineering.py` |
| Saumya | `src/agent_graph.py`, `src/decision_engine.py`, `src/prompts.py` |
| Dikshit | `pages/1_Dashboard.py`, `pages/2_Forecasting.py`, `pages/3_Promotions.py`, `pages/4_AI_Chat.py`, `pages/5_Model_Performance.py` |

---

## 📅 Day-by-Day Plan

---

### ✅ DAY 1 — Set Up & Get Started
**Main goal: Everyone has something running by end of day. No one is blocked.**

#### 🕘 Morning (09:00 – 10:30) — All Together
- All 5 team members meet, clone the GitHub repo, and install all packages
- Parvat walks everyone through this plan and the project structure
- Everyone agrees on the rules (no changing shared functions without asking Parvat)

---

#### 🕙 Afternoon (10:30 – 18:00) — Split and Work

| Person | What They Do Today | What They Deliver by End of Day |
|--------|-------------------|--------------------------------|
| **Parvat** | Sets up the GitHub repository with proper branch structure. Creates the shared `config.py` file that everyone imports. Writes the `setup.sh` script so anyone can install the project in one command. Starts the system architecture diagram. | Repo is live. Everyone can clone and run it. `config.py` is ready. |
| **Himanshu** | Opens `train.csv` and `store.csv` and explores them. Starts writing the data cleaning script. Creates the SQLite database structure. **Must also create a fake (mock) version of the data** so Saumya and Dikshit don't have to wait. | SQLite database schema created. `data/mocks/mock_store_metrics.json` file committed. |
| **Ashutosh** | Studies the data features. Writes the feature engineering script (creates lag columns like "sales 7 days ago"). Runs a simple moving average as the first baseline forecast. **Must also create fake forecast outputs** so Saumya and Dikshit don't have to wait. | `feature_engineering.py` ready. `data/mocks/mock_forecast.json` and `mock_shap.json` committed. |
| **Saumya** | Connects to the LLM (via OpenRouter). Builds the skeleton structure of the LangGraph agent. Uses the mock files from Himanshu and Ashutosh to wire up a basic query that returns text. | LangGraph skeleton compiles. Running a mock query returns a structured response. |
| **Dikshit** | Creates the Streamlit multi-page app shell. Puts placeholder charts on all 5 pages using mock data. Makes sure all the navigation between pages works. | App runs on `localhost:8501` with all 5 pages visible and placeholder content. |

#### 🕕 End of Day Sync (18:00 – 18:15) — All Together
- Everyone pushes their branch to GitHub
- Himanshu and Ashutosh confirm mock files are uploaded
- Parvat checks: can the app import all modules without crashing?

---

### ✅ DAY 2 — Build the Core (Main Work Day)
**Main goal: All real functionality is built. Mocks get replaced with real data.**

| Person | What They Do Today | What They Deliver by End of Day |
|--------|-------------------|--------------------------------|
| **Parvat** | Reviews and merges Pull Requests from Himanshu and Ashutosh. Writes the first version of the presentation stress test. Helps anyone who is blocked. Starts wiring real data into the main `app.py`. | PRs merged. First real end-to-end data flow working. |
| **Himanshu** | Completes **all 5 database query functions** that others will call. Finishes the full EDA notebook — must include: sales trends, store rankings, weekday/weekend patterns, holiday analysis, promo vs non-promo, store type comparison, competition analysis, store clustering, anomaly detection. | All database functions work with real data. EDA notebook has 9+ complete analyses. |
| **Ashutosh** | Trains the full XGBoost 7-day forecasting model using walk-forward cross-validation. Implements SHAP explainability (bar chart + waterfall). Saves the model file. Computes confidence interval bounds. | Model trained, saved, and predicts for any store. SHAP values computed. RMSPE score documented. |
| **Saumya** | Builds all 4 agent nodes in LangGraph (Router → Data Analyst → Forecast → Decision). Switches from mock files to the real Himanshu and Ashutosh functions. Writes the first version of the system prompt in `prompts.py`. | Agent correctly routes queries and calls the right tools with real data. |
| **Dikshit** | Replaces all mock data on the **Dashboard page** (KPI cards, sales trend chart, store ranking table, store type breakdown). Builds the **Forecasting page** (store dropdown, 7-day line chart with confidence bands, SHAP bar chart). | Dashboard and Forecasting pages fully functional with live data. |

#### 🕕 End of Day Sync (18:00 – 18:15) — All Together
- Himanshu and Ashutosh do a live demo of their functions to the team
- Saumya switches from mocks to real functions
- Parvat runs this test live: *"How is Store 100 performing?"* through the agent

---

### ✅ DAY 3 — Connect Everything & Add Advanced Features
**Main goal: The full system works end-to-end. Advanced (S-grade) features are added.**

| Person | What They Do Today | What They Deliver by End of Day |
|--------|-------------------|--------------------------------|
| **Parvat** | Runs the full curveball presentation test (all 5 stores). Fixes any connection problems between modules. Writes shared test fixtures. Finalises the system architecture diagram. | Full end-to-end curveball response works correctly. Architecture diagram done. |
| **Himanshu** | Speeds up database queries with proper indexing. Adds the **promotion uplift ranking** query and **competition impact** analysis. Writes all database tests. Finalises the data assumptions document. | All database tests pass. Promotion analysis queries complete. `docs/data_assumptions.md` done. |
| **Ashutosh** | Fine-tunes XGBoost for better accuracy. Adds the **What-If forecast function** (what happens to sales if we add/remove a promo?). Adds the LightGBM model for comparison. Writes all model tests. Finalises the model report. | Tuned model. What-If function works. All model tests pass. `docs/model_report.md` done. |
| **Saumya** | Completes the **Decision Engine** — every query now returns all 4 sections: Observation, Prediction, Evidence, Recommendation — all with real numbers. Adds **citation list** to every response. Implements the multi-store comparison report. Writes all agent tests. | Decision Engine fully working with real data. All agent tests pass. `docs/agent_design.md` done. |
| **Dikshit** | Builds the **Promotion Analysis page** (uplift ranking chart, store type breakdown, promo timeline). Builds the **AI Assistant page** (streaming chat, citation cards, example question chips). Builds the **Model Performance page** (baseline vs XGBoost chart, RMSPE table). | All 5 pages complete with live data. |

#### 🕕 End of Day Sync (18:00 – 18:30) — Full System Demo — **Critical**
- Parvat runs the curveball question live: *"I manage Stores 100, 200, 300, 400, 500. Which should I focus on next week?"*
- All 5 team members review the output for correctness
- Parvat creates a bug list and assigns fixes to Day 4

---

### ✅ DAY 4 — Polish, Test & Present
**Main goal: Zero crashes. Presentation rehearsed. S-grade delivered.**

| Person | Morning (09:00 – 14:00) | Afternoon (14:00 – 17:00) |
|--------|------------------------|--------------------------|
| **Parvat** | Runs the full test suite (`pytest`). Fixes any failing tests. Runs the curveball test 20 times and verifies consistency. Finalises `README.md`. | Leads the full 15-minute presentation rehearsal with all 5 members. |
| **Himanshu** | Fixes any EDA/database issues from Day 3 bug list. Polishes the EDA notebook (clean output, add written explanations in each cell). | Prepares and rehearses their "Dataset & EDA" presentation section (4 minutes). |
| **Ashutosh** | Validates RMSPE score is documented clearly. Writes the SHAP interpretation guide for a non-technical audience. | Prepares and rehearses their "ML Approach" presentation section (2 minutes). |
| **Saumya** | Runs the agent 20 times with the curveball question. Checks every response for hallucinated numbers. Fixes any prompt issues. | Prepares and rehearses their "Agentic AI & Decision Intelligence" section (3 minutes). |
| **Dikshit** | Final UI polish: adds loading spinners, error messages, empty state screens. Tests all export buttons (CSV download). | Prepares and rehearses the **live application demo** section (3 minutes). |
| **17:00 – 18:00** | **Full 15-minute dry run — all 5 members present in order, just like the real thing.** | |

---

## 🔁 How the System Works Together

Here's the flow when a user types a question in the app:

```
User types question
       ↓
  app.py (Parvat)
       ↓
  agent_graph.py (Saumya) ← understands the question and picks the right tools
       ↓              ↓
  database.py      model_engine.py
   (Himanshu)           (Ashutosh)
  real sales      7-day forecast
  data from       + SHAP values
  SQLite
       ↓              ↓
  decision_engine.py (Saumya)
  builds the structured answer
       ↓
  Back to app.py → shown to user in the AI chat window (Dikshit)
```

---

## 🤝 How the Team Collaborates

### Git Branches (one per person)
```
main               ← Final working code. Parvat approves all merges.
feature/data       ← Himanshu works here
feature/ml         ← Ashutosh works here
feature/agents     ← Saumya works here
feature/ui         ← Dikshit works here
```

### The Mock Rule (prevents Day 1 blocking)
- Himanshu and Ashutosh must publish fake ("mock") output files by **3:00 PM on Day 1**
- Saumya and Dikshit use these mocks immediately so they can start building
- Once real functions are ready, mocks are swapped out with one config change

### Changing a Shared Function — Always Ask First
If any developer needs to change a function that someone else is already calling:
1. Stop coding and message the team immediately
2. Show the old and new version side by side
3. Wait for Parvat to say it's OK and tell the others what to update

### Daily Check-In (15 minutes, end of each day)
Every developer answers three questions:
1. ✅ What did I finish today?
2. 🚧 What am I stuck on?
3. 📤 What do I need from someone else tomorrow?

---

## 🧪 Tests — What Gets Tested and By Whom

| Test File | Written By | What It Checks |
|-----------|-----------|----------------|
| `test_data_pipeline.py` | Himanshu | Data loads correctly, no nulls after cleaning, lag features have no data leakage |
| `test_database.py` | Himanshu | All query functions return correct column names and data types |
| `test_model_engine.py` | Ashutosh | Forecast has exactly 7 rows, all values are positive, SHAP returns top 5 features |
| `test_agent_graph.py` | Saumya | Agent completes without crashing, response contains all 4 sections |
| `test_decision_engine.py` | Saumya | Decision report has all required fields, risk level is valid, citations are non-empty |
| `test_curveball.py` | Parvat | The live presentation question runs 20 times without crashing, same store ranked #1 consistently |

---

## 🎤 Presentation Breakdown

| Speaking Order | Section | Who Presents | How Long |
|---------------|---------|-------------|---------|
| 1 | Business problem and dataset overview | Himanshu | 2 minutes |
| 2 | Data cleaning decisions and key EDA findings | Himanshu | 2 minutes |
| 3 | Forecasting model and accuracy results | Ashutosh | 2 minutes |
| 4 | AI agent design and Decision Intelligence | Saumya | 3 minutes |
| 5 | Live application demo | Dikshit | 3 minutes |
| 6 | System architecture and technical choices | Parvat | 2 minutes |
| 7 | Q&A and curveball question | Parvat leads | Open |

---

## 🎯 The Curveball Question — How We'll Handle It

> *"I manage Stores 100, 200, 300, 400, and 500. Based on historical performance and your forecast, which stores should I focus on next week, why, and what does their promotional history tell me?"*

**What happens in the app automatically:**
1. The agent reads the question and identifies 5 stores
2. It fetches 30-day historical sales for all 5 stores
3. It fetches the 7-day forecast for all 5 stores
4. It fetches the promo history and uplift score for all 5 stores
5. It ranks the stores by risk (worst forecast + declining trend = top priority)
6. It generates a clear answer with all 4 sections:

> **Observation:** Store 200 has seen a 18% sales decline over the last 4 weeks.
> **Prediction:** Our model forecasts continued weakness next week (−12% vs fleet average).
> **Evidence:** SHAP analysis shows promotional activity drives 31% uplift for this store on promo days — but it has had 0 promo days in the past 3 weeks.
> **Recommendation:** Activate Promo 1 for Store 200 next week. Historical data shows this store responds strongly to promotions, and the timing aligns with a typically strong trading period.

---

## ⚠️ Rules Everyone Must Follow

| Rule | Reason |
|------|--------|
| Only use the shared database functions in `database.py` — never write your own SQL queries in `app.py` or `pages/` | Keeps all data access in one place; Himanshu controls it |
| Only use the shared model functions in `model_engine.py` — never load the `.pkl` file directly | Keeps all ML logic in one place; Ashutosh controls it |
| The AI must never make up numbers — all figures must come from actual tool results | Prevents hallucination; the evaluators will check this |
| Never hardcode a store ID like `store_id = 100` | Everything must work dynamically for any store |
| Always import paths from `config.py`, never type file paths directly | Makes the project portable and consistent |
| When building lag features, always use `.shift()` on the correct column | Prevents data leakage — using future data to predict the past, which is a critical ML error |
| Remove all rows where `Open == 0` before training the model | A closed store with 0 sales should not teach the model anything |

---

## 🏁 How We Know We're Done — Per Person

### Parvat is done when:
- All modules connect and the full app runs without errors
- All 50+ tests pass
- Curveball test runs 20 times with zero crashes and consistent answers
- Presentation has been rehearsed by all 5 people

### Himanshu is done when:
- `data_pipeline.py` runs end-to-end and creates a working `retail.db`
- All 8 database functions return correct real data
- EDA notebook has 9+ complete, well-written analyses
- All data tests pass
- Data assumptions document is written

### Ashutosh is done when:
- The XGBoost model predicts 7 days for any store
- RMSPE score is calculated and documented
- SHAP waterfall and bar chart data works for any store
- The What-If forecast changes output when a promo is toggled
- All model tests pass
- Model report is written

### Saumya is done when:
- Any natural language question returns a structured response
- Decision Engine produces all 4 sections (Observation, Prediction, Evidence, Recommendation) using real numbers
- The curveball multi-store comparison works correctly
- All agent and decision tests pass
- Agent design document is written

### Dikshit is done when:
- All 5 Streamlit pages show real (not mock) data
- No page crashes during a 30-minute user test
- AI chat streams responses and shows citation cards
- CSV export works on the forecast and dashboard pages
- The app looks polished — no broken layouts or missing charts

---

## 📊 At-a-Glance Summary Table

|  | DAY 1 | DAY 2 | DAY 3 | DAY 4 |
|--|-------|-------|-------|-------|
| **Parvat** | Project setup, GitHub, config | Review PRs, wire real data | Curveball test, architecture diagram | Full test suite, presentation rehearsal |
| **Himanshu** | Data profiling, DB schema, mock files | All DB functions, full EDA notebook | DB performance, promo & competition analysis, write tests | Fix issues, polish EDA, rehearse |
| **Ashutosh** | Feature engineering, mock files | Train XGBoost + SHAP, save model | Tune model, What-If function, write tests | Validate metrics, write docs, rehearse |
| **Saumya** | LLM setup, LangGraph skeleton | All agent nodes with real data, system prompt | Full Decision Engine, multi-store comparison, write tests | 20× stress test, fix prompts, rehearse |
| **Dikshit** | App shell with all 5 pages (mock) | Dashboard + Forecasting pages (live data) | Promo, Chat, Model Performance pages (live data) | UI polish, export buttons, rehearse |
| **End of Day Goal** | Mocks published. App boots. | First real data flows end-to-end. | Full curveball works live. | Zero crashes. Presentation ready. |

---


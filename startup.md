# 🚀 Getting Started & Quick Startup Guide

Follow these steps to set up and run the **AI-Powered Retail Sales Forecasting & Decision Intelligence** platform on your local machine.

---

## 📋 Quick Start Instructions

Run the following commands in your terminal:

```bash
git clone https://github.com/parvatkhattak/AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence.git
cd AI-Powered-Retail-Sales-Forecasting-Decision-Intelligence
bash setup.sh
```

---

## 🛠️ What `setup.sh` Does Automatically

1. **Creates Virtual Environment**: Initializes a python virtual environment (`venv`).
2. **Installs Dependencies**: Installs all required packages from `requirements.txt`.
3. **Configures Environment Variables**: Creates `.env` from `.env.example` if it doesn't already exist.
4. **Prepares Data Directories**: Sets up local storage directories for data and SQLite database files.

---

## ▶️ Running the Application

Once setup completes, activate the virtual environment (if not already active) and start Streamlit:

```bash
source venv/bin/activate
streamlit run app.py
```

Open your browser at `http://localhost:8501` to view the dashboard!

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MOCKS_DIR  = DATA_DIR / "mocks"
DOCS_DIR   = ROOT_DIR / "docs"

DB_PATH          = DATA_DIR / "retail.db"
MODEL_PATH       = MODELS_DIR / "xgboost_model.pkl"
LGBM_PATH        = MODELS_DIR / "lgbm_model.pkl"
SHAP_PATH        = MODELS_DIR / "shap_explainer.pkl"

TRAIN_CSV        = DATA_DIR / "train.csv"
STORE_CSV        = DATA_DIR / "store.csv"

# ── Mock files (used when USE_MOCKS = True) ───────────────────────────────────
MOCK_STORE_METRICS = MOCKS_DIR / "mock_store_metrics.json"
MOCK_FORECAST      = MOCKS_DIR / "mock_forecast.json"
MOCK_SHAP          = MOCKS_DIR / "mock_shap.json"

# ── Dev flags ─────────────────────────────────────────────────────────────────
USE_MOCKS = False          # Flip to True on Day 1 before real data is ready

# ── LLM Settings ──────────────────────────────────────────────────────────────
LLM_PROVIDER   = "openrouter"                          # "openrouter" | "ollama"
LLM_MODEL      = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")   # Free tier — no billing needed, overridable via env
LLM_BASE_URL   = "https://openrouter.ai/api/v1"
LLM_API_KEY    = os.getenv("OPENROUTER_API_KEY", "")
LLM_TEMPERATURE = 0                                    # Deterministic = no invented numbers

# ── Model Hyperparameters ──────────────────────────────────────────────────────
FORECAST_DAYS   = 7
LAG_DAYS        = [7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]
N_CLUSTERS      = 4          # KMeans clusters for store grouping
ANOMALY_ZSCORE  = 2.5        # Z-score threshold for anomaly flagging

# Walk-forward CV: number of rolling FORECAST_DAYS-length validation folds,
# each with an expanding training window (see train_model()). 6 folds x 7 days
# = same 42-day holdout region the old single-split evaluation used, just
# validated properly instead of as one block.
WALKFORWARD_FOLDS = 6

# ── App Settings ──────────────────────────────────────────────────────────────
APP_TITLE       = "Retail AI — Decision Intelligence Platform"
APP_ICON        = "🛒"
APP_LAYOUT      = "wide"

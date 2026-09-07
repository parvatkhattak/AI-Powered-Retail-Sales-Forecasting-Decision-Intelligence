"""
src/database.py
Owner: Dev 2 — Data Engineer

Responsibilities:
- All read access to the SQLite retail.db
- Expose data via contract functions for other modules
- Fallback to mock data if USE_MOCKS is True
"""

import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import USE_MOCKS, MOCK_STORE_METRICS, DB_PATH

def get_store_metrics(store_ids: list[int], days: int = 30) -> pd.DataFrame:
    """Returns daily sales, customers, promo flag for given stores over last N days."""
    if USE_MOCKS:
        # Dev 2: Replace this with real DB query logic when ready
        print("MOCK: Returning mock store metrics")
        return pd.DataFrame([{"store_id": 100, "sales": 5000, "promo": 1}])
    
    # TODO (Dev 2): Write real SQLAlchemy query here
    return pd.DataFrame()

def get_promo_history(store_id: int) -> dict:
    """Returns promo vs non-promo sales summary for a store."""
    if USE_MOCKS:
        return {"promo_avg_sales": 6000, "non_promo_avg_sales": 4000, "uplift_pct": 50.0}
    
    # TODO (Dev 2): Write real SQLAlchemy query here
    return {}

def get_eda_summary() -> dict:
    """Returns top-level KPIs: total stores, avg daily sales, best/worst performers."""
    # TODO (Dev 2): Implement
    return {}

def get_sales_trend(store_id: int, period: str = "monthly") -> pd.DataFrame:
    """Returns aggregated sales trend for a store."""
    # TODO (Dev 2): Implement
    return pd.DataFrame()

def get_all_stores() -> pd.DataFrame:
    """Full store list with StoreType, Assortment, CompetitionDistance."""
    # TODO (Dev 2): Implement
    return pd.DataFrame()

def get_promo_uplift_ranking(top_n: int = 10) -> pd.DataFrame:
    """Stores ranked by promotional sales uplift percentage."""
    # TODO (Dev 2): Implement
    return pd.DataFrame()

def get_store_cluster(store_id: int) -> dict:
    """Returns cluster_id and cluster_label for the store."""
    # TODO (Dev 2): Implement (S-Grade)
    return {}

def get_anomaly_flags(store_id: int, lookback_days: int = 90) -> pd.DataFrame:
    """Returns dates flagged as anomalous sales."""
    # TODO (Dev 2): Implement (S-Grade)
    return pd.DataFrame()

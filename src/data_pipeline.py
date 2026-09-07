"""
src/data_pipeline.py
Owner: Himanshu — Data Engineer

Responsibilities:
- Load and merge train.csv + store.csv
- Clean and resolve data quality issues
- Engineer all features (lags, rolling, calendar, competition, promo)
- Populate the SQLite database (retail.db)

Run this script once before starting the app:
    python src/data_pipeline.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, TRAIN_CSV, STORE_CSV, LAG_DAYS, ROLLING_WINDOWS, N_CLUSTERS, ANOMALY_ZSCORE


# ── 1. Load Raw Data ───────────────────────────────────────────────────────────

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and store.csv from disk."""
    print("📂 Loading raw data...")
    train = pd.read_csv(TRAIN_CSV, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(STORE_CSV)
    print(f"   train.csv: {len(train):,} rows | store.csv: {len(store):,} rows")
    return train, store


# ── 2. Merge ───────────────────────────────────────────────────────────────────

def merge_datasets(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Left-join train on store using Store column."""
    # TODO (Himanshu): Implement merge and validate row count
    df = train.merge(store, on="Store", how="left")
    assert len(df) == len(train), "Row count mismatch after merge!"
    print(f"   ✅ Merged: {len(df):,} rows")
    return df


# ── 3. Clean ───────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve all data quality issues:
    - Remove rows where Open == 0 (closed store days)
    - Fill CompetitionDistance nulls with median
    - Normalise StateHoliday types
    - Encode PromoInterval
    """
    # TODO (Himanshu): Implement full cleaning logic per data_assumptions.md
    df = df[df["Open"] == 1].copy()
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "no_holiday")
    print(f"   ✅ Cleaned: {len(df):,} rows (removed closed-store days)")
    return df


# ── 4. Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all model features. Delegates to feature_engineering.py.
    """
    # TODO (Himanshu + Ashutosh): Call feature_engineering functions here
    from src.feature_engineering import (
        add_date_features,
        add_lag_features,
        add_rolling_features,
        add_competition_features,
        add_promo_features,
    )
    df = add_date_features(df)
    df = add_lag_features(df, lag_days=LAG_DAYS)
    df = add_rolling_features(df, windows=ROLLING_WINDOWS)
    df = add_competition_features(df)
    df = add_promo_features(df)
    print(f"   ✅ Features engineered: {df.shape[1]} columns")
    return df


# ── 5. Store Clustering ────────────────────────────────────────────────────────

def assign_store_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster stores by sales behaviour using KMeans."""
    # TODO (Himanshu): Implement KMeans clustering on aggregated store features
    print(f"   ✅ Store clusters assigned (K={N_CLUSTERS})")
    return df


# ── 6. Anomaly Detection ──────────────────────────────────────────────────────

def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag daily sales rows with Z-score > ANOMALY_ZSCORE as anomalous."""
    # TODO (Himanshu): Implement Z-score anomaly flagging per store
    df["is_anomaly"] = 0
    print(f"   ✅ Anomaly flags added (threshold z={ANOMALY_ZSCORE})")
    return df


# ── 7. Write to SQLite ─────────────────────────────────────────────────────────

def write_to_sqlite(df: pd.DataFrame) -> None:
    """Write cleaned + featured data to SQLite database."""
    # TODO (Himanshu): Write sales and stores tables; add indexes
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("sales", engine, if_exists="replace", index=False)
    print(f"   ✅ Written to {DB_PATH}")
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_store_date ON sales (Store, Date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_store ON sales (Store)"))
        conn.commit()
    print("   ✅ Indexes created")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Run the full data pipeline end-to-end."""
    print("\n🚀 Starting data pipeline...\n")
    train, store = load_raw_data()
    df = merge_datasets(train, store)
    df = clean_data(df)
    df = engineer_features(df)
    df = assign_store_clusters(df)
    df = flag_anomalies(df)
    write_to_sqlite(df)
    print("\n✅ Pipeline complete! retail.db is ready.\n")


if __name__ == "__main__":
    run_pipeline()

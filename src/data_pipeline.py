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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    df = train.merge(store, on="Store", how="left", validate="many_to_one")
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
    
    # Store original row count before cleaning
    before = len(df)
    df = df[df["Open"] == 1].copy()
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)
    
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
        if col in df.columns:
                df[col] = df[col].fillna(0)
                
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace(
        {"0": "no_holiday", "0.0": "no_holiday"}
    )
    if "PromoInterval" in df.columns and "Date" in df.columns:
        df["PromoInterval"] = (
            df["PromoInterval"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        month_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        row_month_name = df["Date"].dt.month.map(month_map)
        df["IsPromoMonth"] = [
            int(
                m in [x.strip() for x in interval.split(",")]
            )
            for m, interval in zip(
                row_month_name,
                df["PromoInterval"]
            )
        ]
    else:
        df["IsPromoMonth"] = 0
 
    print(f"   ✅ Cleaned: {len(df):,} rows (removed {before - len(df):,} closed-store days)")
    
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
    n_clusters = N_CLUSTERS
 
    store_features = (
        df.groupby("Store")
        .agg(
            avg_sales=("Sales", "mean"),
            avg_customers=("Customers", "mean") if "Customers" in df.columns else ("Sales", "mean"),
            sales_std=("Sales", "std"),
            promo_frequency=("Promo", "mean") if "Promo" in df.columns else ("Sales", "mean"),
        )
        .fillna(0)
        .reset_index()
    )
 
    feature_cols = ["avg_sales", "avg_customers", "sales_std", "promo_frequency"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(store_features[feature_cols])
 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    store_features["store_cluster"] = kmeans.fit_predict(scaled)
 
    df = df.merge(store_features[["Store", "store_cluster"]], on="Store", how="left")
    print(f"   ✅ Store clusters assigned (K={N_CLUSTERS})")
    return df


# ── 6. Anomaly Detection ──────────────────────────────────────────────────────

def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag daily sales rows with Z-score > ANOMALY_ZSCORE as anomalous."""
    z_threshold = ANOMALY_ZSCORE
 
    store_mean = df.groupby("Store")["Sales"].transform("mean")
    store_std = df.groupby("Store")["Sales"].transform("std").replace(0, np.nan)
 
    z_scores = (df["Sales"] - store_mean) / store_std
    df["sales_zscore"] = z_scores.fillna(0)
    df["is_anomaly"] = (df["sales_zscore"].abs() > z_threshold).astype(int)
 
    n_flagged = int(df["is_anomaly"].sum())
    print(f"   ✅ Anomaly flags added (threshold z={z_threshold}) — {n_flagged:,} rows flagged")
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

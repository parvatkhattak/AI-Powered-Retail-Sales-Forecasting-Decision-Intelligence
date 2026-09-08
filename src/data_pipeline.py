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
import plotly.express as px
import plotly.graph_objects as go
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
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
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

# ── 7. Create Visualizations ──────────────────────────────────────────────────
def create_visualizations(df: pd.DataFrame) -> None:
    """
    Create EDA visualizations directly from the processed DataFrame.
 
    Visualizations are generated from the CSV data after cleaning,
    feature engineering, clustering and anomaly detection.
 
    No API calls are used.
    """
 
    print("\n📊 Creating visualizations...")
 
    # Create output directory
    visualization_dir = Path(__file__).resolve().parent.parent / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
 
    # Only analyze open stores (defensive: "Open" may have been dropped as a
    # constant column by feature_engineering.py, since every remaining row is
    # already Open == 1 by this point in the pipeline)
    analysis_df = df[df["Open"] == 1].copy() if "Open" in df.columns else df.copy()
 
    # ────────────────────────────────────────────────────────────────────────
    # 1. Daily Sales
    # ────────────────────────────────────────────────────────────────────────
 
    daily_sales = (
        analysis_df
        .groupby("Date", as_index=False)
        .agg(
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
        )
        .sort_values("Date")
    )
 
    fig = px.line(
        daily_sales,
        x="Date",
        y="total_sales",
        title="Daily Total Sales",
        labels={
            "Date": "Date",
            "total_sales": "Total Sales",
        },
    )
 
    fig.update_layout(hovermode="x unified")
 
    fig.write_image(
        visualization_dir / "01_daily_sales.png"
    )
       # ────────────────────────────────────────────────────────────────────────
    # 2. Monthly Sales
    # ────────────────────────────────────────────────────────────────────────
 
    monthly_sales = (
        analysis_df
        .assign(
            month=analysis_df["Date"]
            .dt
            .to_period("M")
            .astype(str)
        )
        .groupby("month", as_index=False)
        .agg(
            total_sales=("Sales", "sum"),
            avg_sales=("Sales", "mean"),
        )
        .sort_values("month")
    )
 
    fig = px.bar(
        monthly_sales,
        x="month",
        y="total_sales",
        title="Monthly Total Sales",
        labels={
            "month": "Month",
            "total_sales": "Total Sales",
        },
    )
 
    fig.write_image(
        visualization_dir / "02_monthly_sales.png"
    )
    # ────────────────────────────────────────────────────────────────────────
    # 3. Day of Week Sales
    # ────────────────────────────────────────────────────────────────────────
 
    weekday_sales = (
        analysis_df
        .groupby("DayOfWeek", as_index=False)
        .agg(
            avg_sales=("Sales", "mean"),
            total_sales=("Sales", "sum"),
            avg_customers=("Customers", "mean"),
        )
        .sort_values("DayOfWeek")
    )
 
    day_names = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }
 
    weekday_sales["day_name"] = (
        weekday_sales["DayOfWeek"].map(day_names)
    )
 
    fig = px.bar(
        weekday_sales,
        x="day_name",
        y="avg_sales",
        title="Average Sales by Day of Week",
        labels={
            "day_name": "Day",
            "avg_sales": "Average Sales",
        },
    )
 
    fig.write_image(
        visualization_dir / "03_weekday_sales.png"
    )
 
    # ────────────────────────────────────────────────────────────────────────
    # 4. Store Performance
    # ────────────────────────────────────────────────────────────────────────
 
    store_performance = (
        analysis_df
        .groupby("Store", as_index=False)
        .agg(
            avg_sales=("Sales", "mean"),
            total_sales=("Sales", "sum"),
        )
        .sort_values(
            "avg_sales",
            ascending=False
        )
        .head(20)
    )
 
    store_performance["Store"] = (
        store_performance["Store"].astype(str)
    )
 
    fig = px.bar(
        store_performance,
        x="Store",
        y="avg_sales",
        title="Top 20 Stores by Average Daily Sales",
        labels={
            "Store": "Store",
            "avg_sales": "Average Daily Sales",
        },
    )
 
    fig.write_image(
        visualization_dir / "04_store_performance.png"
    )
    # ────────────────────────────────────────────────────────────────────────
    # 5. Store Type
    # ────────────────────────────────────────────────────────────────────────
 
    store_type = (
        analysis_df
        .groupby("StoreType", as_index=False)
        .agg(
            avg_sales=("Sales", "mean"),
            total_sales=("Sales", "sum"),
            avg_customers=("Customers", "mean"),
        )
    )
 
    fig = px.bar(
        store_type,
        x="StoreType",
        y="avg_sales",
        title="Average Sales by Store Type",
        labels={
            "StoreType": "Store Type",
            "avg_sales": "Average Sales",
        },
    )
 
    fig.write_image(
        visualization_dir / "05_store_type.png"
    )
 
    # ────────────────────────────────────────────────────────────────────────
    # 6. Promotion Analysis
    # ────────────────────────────────────────────────────────────────────────
 
    promo = (
        analysis_df
        .groupby("Promo", as_index=False)
        .agg(
            avg_sales=("Sales", "mean"),
            total_sales=("Sales", "sum"),
            avg_customers=("Customers", "mean"),
        )
    )
 
    promo["promo_label"] = promo["Promo"].map({
        0: "No Promotion",
        1: "Promotion",
    })
 
    fig = px.bar(
        promo,
        x="promo_label",
        y="avg_sales",
        title="Average Sales: Promotion vs No Promotion",
        labels={
            "promo_label": "Promotion Status",
            "avg_sales": "Average Sales",
        },
    )
 
    fig.write_image(
        visualization_dir / "06_promotion.png"
    )
    # ────────────────────────────────────────────────────────────────────────
    # 7. Promotion Uplift
    # ────────────────────────────────────────────────────────────────────────
 
    promo_sales = (
        analysis_df[analysis_df["Promo"] == 1]
        .groupby("Store")["Sales"]
        .mean()
    )
 
    non_promo_sales = (
        analysis_df[analysis_df["Promo"] == 0]
        .groupby("Store")["Sales"]
        .mean()
    )
 
    promo_uplift = pd.DataFrame({
        "promo_avg_sales": promo_sales,
        "non_promo_avg_sales": non_promo_sales,
    }).dropna()
 
    promo_uplift["uplift_pct"] = (
        (
            promo_uplift["promo_avg_sales"]
            - promo_uplift["non_promo_avg_sales"]
        )
        / promo_uplift["non_promo_avg_sales"]
        * 100
    )
 
    promo_uplift = (
        promo_uplift
        .reset_index()
        .sort_values(
            "uplift_pct",
            ascending=False
        )
        .head(20)
    )
 
    promo_uplift["Store"] = (
        promo_uplift["Store"].astype(str)
    )
 
    fig = px.bar(
        promo_uplift,
        x="Store",
        y="uplift_pct",
        title="Top 20 Stores by Promotional Sales Uplift",
        labels={
            "Store": "Store",
            "uplift_pct": "Sales Uplift (%)",
        },
    )
 
    fig.write_image(
        visualization_dir / "07_promo_uplift.png"
    )
 
    # ────────────────────────────────────────────────────────────────────────
    # 8. Assortment
    # ────────────────────────────────────────────────────────────────────────
 
    assortment = (
        analysis_df
        .groupby("Assortment", as_index=False)
        .agg(
            avg_sales=("Sales", "mean"),
            total_sales=("Sales", "sum"),
            avg_customers=("Customers", "mean"),
        )
    )
 
    fig = px.bar(
        assortment,
        x="Assortment",
        y="avg_sales",
        title="Average Sales by Assortment Type",
        labels={
            "Assortment": "Assortment",
            "avg_sales": "Average Sales",
        },
    )
 
    fig.write_image(
        visualization_dir / "08_assortment.png"
    )
 
    # ────────────────────────────────────────────────────────────────────────
    # 9. Holiday Analysis
    # ────────────────────────────────────────────────────────────────────────
 
    holiday = (
        analysis_df
        .groupby("StateHoliday", as_index=False)
        .agg(
            avg_sales=("Sales", "mean"),
            total_sales=("Sales", "sum"),
            avg_customers=("Customers", "mean"),
        )
    )
 
    fig = px.bar(
        holiday,
        x="StateHoliday",
        y="avg_sales",
        title="Average Sales by Holiday Type",
        labels={
            "StateHoliday": "Holiday Type",
            "avg_sales": "Average Sales",
        },
    )
 
    fig.write_image(
        visualization_dir / "09_holiday.png"
    )
 
    # ────────────────────────────────────────────────────────────────────────
    # 10. Competition Distance vs Sales
    # ────────────────────────────────────────────────────────────────────────
 
    competition = (
        analysis_df
        .groupby("Store", as_index=False)
        .agg(
            competition_distance=(
                "CompetitionDistance",
                "first",
            ),
            avg_sales=("Sales", "mean"),
        )
        .dropna()
    )
 
    fig = px.scatter(
        competition,
        x="competition_distance",
        y="avg_sales",
        title="Competition Distance vs Average Sales",
        labels={
            "competition_distance": "Competition Distance",
            "avg_sales": "Average Sales",
        },
        hover_data=["Store"],
    )
 
    fig.write_image(
        visualization_dir / "10_competition.png"
    )
 
    # ────────────────────────────────────────────────────────────────────────
    # 11. Store Clusters
    # ────────────────────────────────────────────────────────────────────────
 
    if "store_cluster" in analysis_df.columns:
 
        clusters = (
            analysis_df
            .groupby("store_cluster", as_index=False)
            .agg(
                avg_sales=("Sales", "mean"),
                avg_customers=("Customers", "mean"),
                store_count=("Store", "nunique"),
            )
        )
 
        fig = px.scatter(
            clusters,
            x="avg_customers",
            y="avg_sales",
            size="store_count",
            text="store_cluster",
            title="Store Clusters",
            labels={
                "avg_customers": "Average Customers",
                "avg_sales": "Average Sales",
                "store_cluster": "Cluster",
            },
        )
 
        fig.write_image(
            visualization_dir / "11_clusters.png"
        )
 
    # ────────────────────────────────────────────────────────────────────────
    # 12. Anomaly Detection
    # ────────────────────────────────────────────────────────────────────────
 
    if "is_anomaly" in analysis_df.columns:
 
        daily = (
            analysis_df
            .groupby("Date", as_index=False)
            .agg(
                total_sales=("Sales", "sum"),
                anomaly_count=("is_anomaly", "sum"),
            )
            .sort_values("Date")
        )
 
        fig = px.line(
            daily,
            x="Date",
            y="total_sales",
            title="Daily Sales and Anomalies",
            labels={
                "Date": "Date",
                "total_sales": "Total Sales",
            },
        )
 
        anomalies = daily[
            daily["anomaly_count"] > 0
        ]
 
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["Date"],
                    y=anomalies["total_sales"],
                    mode="markers",
                    name="Anomalous Days",
                    text=anomalies["anomaly_count"],
                    hovertemplate=(
                        "Date: %{x}<br>"
                        "Sales: %{y}<br>"
                        "Anomalous rows: %{text}"
                        "<extra></extra>"
                    ),
                )
            )
 
        fig.write_image(
            visualization_dir / "12_anomalies.png"
        )
 
    print(
        f"   ✅ Visualizations saved to: "
        f"{visualization_dir}"
    )
    
# ── 8. Write to SQLite ─────────────────────────────────────────────────────────

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
    create_visualizations(df)
    print("\n✅ Pipeline complete! retail.db is ready.\n")


if __name__ == "__main__":
    run_pipeline()

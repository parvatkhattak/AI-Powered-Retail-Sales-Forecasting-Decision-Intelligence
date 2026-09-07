"""
src/database.py
Owner: Himanshu — Data Engineer

Responsibilities:
- All read access to the SQLite retail.db
- Expose data via contract functions for other modules
- Fallback to mock data if USE_MOCKS is True
"""

import pandas as pd
import json
from pathlib import Path
import sys
from sqlalchemy import create_engine,text
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import USE_MOCKS, MOCK_STORE_METRICS, DB_PATH


_engine = None

def _get_engine():
    """Lazily create a single shared SQLAlchemy engine for the SQLite DB."""
    global _engine
    if _engine is None:
        _engine = create_engine(f"sqlite:///{DB_PATH}")
    return _engine

def get_store_metrics(store_ids: list[int], days: int = 30) -> pd.DataFrame:
    """Returns daily sales, customers, promo flag for given stores over last N days."""
    if USE_MOCKS:
        print("MOCK: Returning mock store metrics")
        mock_df = pd.DataFrame(MOCK_STORE_METRICS)
        if "store_id" in mock_df.columns:
            mock_df = mock_df[mock_df["store_id"].isin(store_ids)] if store_ids else mock_df
        return mock_df.reset_index(drop=True)
    
    if not store_ids:
        return pd.DataFrame(columns=["store_id", "date", "sales", "customers", "promo"])
 
    engine = _get_engine()
    placeholders = ", ".join(f":store_{i}" for i in range(len(store_ids)))
    params = {f"store_{i}": sid for i, sid in enumerate(store_ids)}
    params["days"] = days
 
    query = text(f"""
        SELECT
            Store AS store_id,
            Date AS date,
            Sales AS sales,
            Customers AS customers,
            Promo AS promo
        FROM sales
        WHERE Store IN ({placeholders})
          AND Date >= date((SELECT MAX(Date) FROM sales), '-' || :days || ' days')
        ORDER BY Store, Date
    """)
 
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
    return df

def get_promo_history(store_id: int) -> dict:
    """Returns promo vs non-promo sales summary for a store."""
    if USE_MOCKS:
        return {"promo_avg_sales": 6000, "non_promo_avg_sales": 4000, "uplift_pct": 50.0}
    
    engine = _get_engine()
    query = text("""
        SELECT Promo, AVG(Sales) AS avg_sales
        FROM sales
        WHERE Store = :store_id
        GROUP BY Promo
    """)
 
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"store_id": store_id})
 
    if df.empty:
        return {"promo_avg_sales": 0.0, "non_promo_avg_sales": 0.0, "uplift_pct": 0.0}
 
    promo_row = df[df["Promo"] == 1]
    non_promo_row = df[df["Promo"] == 0]
 
    promo_avg = float(promo_row["avg_sales"].iloc[0]) if not promo_row.empty else 0.0
    non_promo_avg = float(non_promo_row["avg_sales"].iloc[0]) if not non_promo_row.empty else 0.0
    uplift_pct = ((promo_avg - non_promo_avg) / non_promo_avg * 100.0) if non_promo_avg > 0 else 0.0
 
    return {
        "promo_avg_sales": round(promo_avg, 2),
        "non_promo_avg_sales": round(non_promo_avg, 2),
        "uplift_pct": round(uplift_pct, 2),
    }

def get_eda_summary() -> dict:
    """Returns top-level KPIs: total stores, avg daily sales, best/worst performers."""
    if USE_MOCKS:
        return {
            "total_stores": 50,
            "avg_daily_sales": 5500.0,
            "best_store_id": 100,
            "best_store_avg_sales": 9000.0,
            "worst_store_id": 200,
            "worst_store_avg_sales": 2000.0,
        }
 
    engine = _get_engine()
 
    overall_query = text("""
        SELECT COUNT(DISTINCT Store) AS total_stores, AVG(Sales) AS avg_daily_sales
        FROM sales
    """)
    per_store_query = text("""
        SELECT Store, AVG(Sales) AS avg_sales
        FROM sales
        GROUP BY Store
        ORDER BY avg_sales DESC
    """)
 
    with engine.connect() as conn:
        overall = pd.read_sql(overall_query, conn)
        per_store = pd.read_sql(per_store_query, conn)
 
    if overall.empty or per_store.empty:
        return {}
 
    best = per_store.iloc[0]
    worst = per_store.iloc[-1]
 
    return {
        "total_stores": int(overall["total_stores"].iloc[0]),
        "avg_daily_sales": round(float(overall["avg_daily_sales"].iloc[0]), 2),
        "best_store_id": int(best["Store"]),
        "best_store_avg_sales": round(float(best["avg_sales"]), 2),
        "worst_store_id": int(worst["Store"]),
        "worst_store_avg_sales": round(float(worst["avg_sales"]), 2),
    }

def get_sales_trend(store_id: int, period: str = "monthly") -> pd.DataFrame:
    """Returns aggregated sales trend for a store."""
    if USE_MOCKS:
        return pd.DataFrame([
            {"period": "2024-01", "total_sales": 150000, "avg_sales": 5000},
            {"period": "2024-02", "total_sales": 160000, "avg_sales": 5300},
        ])
 
    strftime_map = {
        "daily": "%Y-%m-%d",
        "weekly": "%Y-%W",
        "monthly": "%Y-%m",
    }
    fmt = strftime_map.get(period, "%Y-%m")
 
    engine = _get_engine()
    query = text(f"""
        SELECT
            strftime('{fmt}', Date) AS period,
            SUM(Sales) AS total_sales,
            AVG(Sales) AS avg_sales
        FROM sales
        WHERE Store = :store_id
        GROUP BY period
        ORDER BY period
    """)
 
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"store_id": store_id})
    return df

def get_all_stores() -> pd.DataFrame:
    """Full store list with StoreType, Assortment, CompetitionDistance."""
    if USE_MOCKS:
        return pd.DataFrame([
            {"Store": 100, "StoreType": "a", "Assortment": "a", "CompetitionDistance": 1200},
            {"Store": 101, "StoreType": "b", "Assortment": "c", "CompetitionDistance": 850},
        ])
 
    engine = _get_engine()
    query = text("""
        SELECT DISTINCT Store, StoreType, Assortment, CompetitionDistance
        FROM sales
        ORDER BY Store
    """)
 
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_promo_uplift_ranking(top_n: int = 10) -> pd.DataFrame:
    """Stores ranked by promotional sales uplift percentage."""
    if USE_MOCKS:
        return pd.DataFrame([
            {"Store": 100, "promo_avg_sales": 6000, "non_promo_avg_sales": 4000, "uplift_pct": 50.0},
        ])
 
    engine = _get_engine()
    query = text("""
        SELECT
            Store,
            AVG(CASE WHEN Promo = 1 THEN Sales END) AS promo_avg_sales,
            AVG(CASE WHEN Promo = 0 THEN Sales END) AS non_promo_avg_sales
        FROM sales
        GROUP BY Store
        HAVING promo_avg_sales IS NOT NULL AND non_promo_avg_sales IS NOT NULL
               AND non_promo_avg_sales > 0
    """)
 
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
 
    if df.empty:
        return df
 
    df["uplift_pct"] = (
        (df["promo_avg_sales"] - df["non_promo_avg_sales"]) / df["non_promo_avg_sales"] * 100.0
    ).round(2)
    df["promo_avg_sales"] = df["promo_avg_sales"].round(2)
    df["non_promo_avg_sales"] = df["non_promo_avg_sales"].round(2)
 
    return df.sort_values("uplift_pct", ascending=False).head(top_n).reset_index(drop=True)

def get_store_cluster(store_id: int) -> dict:
    """Returns cluster_id and cluster_label for the store."""
    if USE_MOCKS:
        return {"cluster_id": 0, "cluster_label": "High Performer"}
 
    engine = _get_engine()
    query = text("""
        SELECT store_cluster
        FROM sales
        WHERE Store = :store_id
        LIMIT 1
    """)
 
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"store_id": store_id})
 
    if df.empty or pd.isna(df["store_cluster"].iloc[0]):
        return {}
 
    cluster_id = int(df["store_cluster"].iloc[0])
 
    ranking_query = text("""
        SELECT store_cluster, AVG(Sales) AS avg_sales
        FROM sales
        GROUP BY store_cluster
        ORDER BY avg_sales DESC
    """)
    with engine.connect() as conn:
        ranking = pd.read_sql(ranking_query, conn)
 
    ranking = ranking.reset_index(drop=True)
    ranking["rank"] = ranking.index
    n_clusters = len(ranking)
    rank = int(ranking.loc[ranking["store_cluster"] == cluster_id, "rank"].iloc[0])
 
    if n_clusters <= 1:
        label = "Average Performer"
    elif rank == 0:
        label = "High Performer"
    elif rank == n_clusters - 1:
        label = "Low Performer"
    else:
        label = "Mid Performer"
 
    return {"cluster_id": cluster_id, "cluster_label": label}

def get_anomaly_flags(store_id: int, lookback_days: int = 90) -> pd.DataFrame:
    """Returns dates flagged as anomalous sales."""
    if USE_MOCKS:
        return pd.DataFrame([
            {"date": "2024-03-15", "sales": 12000, "sales_zscore": 3.2},
        ])
 
    engine = _get_engine()
    query = text("""
        SELECT Date AS date, Sales AS sales, sales_zscore
        FROM sales
        WHERE Store = :store_id
          AND is_anomaly = 1
          AND Date >= date((SELECT MAX(Date) FROM sales), '-' || :lookback_days || ' days')
        ORDER BY Date DESC
    """)
 
    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn,
            params={"store_id": store_id, "lookback_days": lookback_days},
            parse_dates=["date"],
        )
    return df

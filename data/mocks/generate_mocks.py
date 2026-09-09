"""
data/mocks/generate_mocks.py
Helper script to regenerate all mock JSON files used during development.
Run once: python data/mocks/generate_mocks.py
"""

import json
import random
import math
from pathlib import Path
from datetime import date, timedelta

random.seed(42)
MOCKS_DIR = Path(__file__).resolve().parent

# ── Store metadata ─────────────────────────────────────────────────────────────
STORE_TYPES = ["a", "b", "c", "d"]
ASSORTMENTS = ["a", "b", "c"]
CLUSTER_LABELS = ["High Performer", "Mid Performer", "Low Performer", "Niche Store"]

STORES = []
for i, sid in enumerate([100, 200, 300, 400, 500, 101, 201, 301, 401, 501, 102, 202]):
    STORES.append(
        {
            "Store": sid,
            "StoreType": STORE_TYPES[i % 4],
            "Assortment": ASSORTMENTS[i % 3],
            "CompetitionDistance": round(random.uniform(300, 5000), 1),
            "cluster_label": CLUSTER_LABELS[i % 4],
            "avg_sales": round(random.uniform(3000, 12000), 2),
        }
    )

# ── Daily metrics (30 days × 6 stores) ───────────────────────────────────────
BASE_DATE = date(2025, 8, 10)
STORE_IDS = [100, 200, 300, 400, 500, 300]
BASE_SALES = {100: 6200, 200: 3100, 300: 8800, 400: 5500, 500: 4100}

metrics = []
for days_back in range(30, 0, -1):
    d = BASE_DATE + timedelta(days=30 - days_back)
    for sid in [100, 200, 300, 400, 500]:
        base = BASE_SALES[sid]
        promo = 1 if random.random() < 0.35 else 0
        noise = random.uniform(0.88, 1.12)
        sales = round(base * noise * (1.25 if promo else 1.0))
        customers = round(sales / random.uniform(9, 13))
        metrics.append(
            {
                "store_id": sid,
                "date": str(d),
                "sales": sales,
                "customers": customers,
                "promo": promo,
            }
        )

(MOCKS_DIR / "mock_store_metrics.json").write_text(json.dumps(metrics, indent=2))

# ── EDA summary ───────────────────────────────────────────────────────────────
eda = {
    "total_stores": 1115,
    "avg_daily_sales": 6956.32,
    "total_sales": 958185660.0,
    "best_store_id": 562,
    "best_store_avg_sales": 22098.44,
    "worst_store_id": 307,
    "worst_store_avg_sales": 812.67,
}
(MOCKS_DIR / "mock_eda_summary.json").write_text(json.dumps(eda, indent=2))

# ── Sales trend (monthly, store 100) ─────────────────────────────────────────
trend = []
for mo in range(1, 13):
    trend.append(
        {
            "period": f"2024-{mo:02d}",
            "total_sales": round(random.uniform(130000, 210000)),
            "avg_sales": round(random.uniform(4500, 7000)),
        }
    )
(MOCKS_DIR / "mock_sales_trend.json").write_text(json.dumps(trend, indent=2))

# ── Promo uplift ranking ──────────────────────────────────────────────────────
uplift_rows = []
for s in STORES[:10]:
    promo_avg = round(s["avg_sales"] * random.uniform(1.1, 1.6), 2)
    non_promo_avg = round(s["avg_sales"] * random.uniform(0.75, 0.95), 2)
    uplift_pct = round((promo_avg - non_promo_avg) / non_promo_avg * 100, 2)
    uplift_rows.append(
        {
            "Store": s["Store"],
            "promo_avg_sales": promo_avg,
            "non_promo_avg_sales": non_promo_avg,
            "uplift_pct": uplift_pct,
            "store_type": s["StoreType"],
        }
    )
uplift_rows.sort(key=lambda r: r["uplift_pct"], reverse=True)
(MOCKS_DIR / "mock_promo_uplift.json").write_text(json.dumps(uplift_rows, indent=2))

# ── Anomaly flags ──────────────────────────────────────────────────────────────
anomalies = [
    {"date": "2025-08-14", "sales": 14500, "sales_zscore": 3.1},
    {"date": "2025-08-21", "sales": 1800,  "sales_zscore": -2.7},
    {"date": "2025-08-28", "sales": 15200, "sales_zscore": 3.4},
]
(MOCKS_DIR / "mock_anomalies.json").write_text(json.dumps(anomalies, indent=2))

print("✅ All mock JSON files regenerated.")

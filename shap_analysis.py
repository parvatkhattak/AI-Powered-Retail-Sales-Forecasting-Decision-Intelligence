import pandas as pd
import numpy as np
import sqlite3
import joblib
import shap
import matplotlib.pyplot as plt

DB_PATH = "data/retail.db"
STATE_HOLIDAY_MAP = {"no_holiday": 0, "a": 1, "b": 2, "c": 3}
STORE_TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP = {"a": 0, "b": 1, "c": 2}
DROP_COLS = ["Sales", "Customers", "Date", "Open", "sales_zscore", "is_anomaly"]

print("Loading model + data...")
model = joblib.load("models/lgbm_model.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
medians = joblib.load("models/impute_medians.pkl")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM sales", conn, parse_dates=["Date"])
conn.close()

df["StateHoliday"] = df["StateHoliday"].map(STATE_HOLIDAY_MAP)
df["StoreType"] = df["StoreType"].map(STORE_TYPE_MAP)
df["Assortment"] = df["Assortment"].map(ASSORTMENT_MAP)

lag_roll_cols = [c for c in feature_cols if c.startswith("Sales_lag_") or c.startswith("Sales_roll_")]
df = df.dropna(subset=lag_roll_cols)
remaining_na_cols = ["CompetitionDistance", "CompetitionOpenMonths", "DaysSinceLastPromo", "DaysUntilNextPromo"]
df[remaining_na_cols] = df[remaining_na_cols].fillna(medians)

# Sample for speed — SHAP is expensive to compute row-by-row
sample = df.sample(n=min(5000, len(df)), random_state=42)
X_sample = sample[feature_cols]

print(f"Computing SHAP values on {len(X_sample):,} sampled rows...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# --- Global importance: mean |SHAP value| per feature ---
importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].barh(importance.index[-15:], importance.values[-15:], color='steelblue')
axes[0].set_title("Top 15 Features by Mean |SHAP Value|\n(overall impact on predictions)")
axes[0].set_xlabel("Mean |SHAP value| (log-sales impact)")

# --- Beeswarm-style summary (SHAP's own plot) ---
plt.sca(axes[1])
shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, max_display=15)
axes[1].set_title("SHAP Summary — direction + magnitude per feature")

plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=110, bbox_inches='tight')
print("\nSaved shap_feature_importance.png")

print("\nTop 10 features by importance:")
print(importance.sort_values(ascending=False).head(10).to_string())

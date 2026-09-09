import pandas as pd
import numpy as np
import sqlite3
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

DB_PATH = "data/retail.db"

STATE_HOLIDAY_MAP = {"no_holiday": 0, "a": 1, "b": 2, "c": 3}
STORE_TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP = {"a": 0, "b": 1, "c": 2}
DROP_COLS = ["Sales", "Customers", "Date", "Open", "sales_zscore", "is_anomaly"]


def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


print("Loading data...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM sales", conn, parse_dates=["Date"])
conn.close()

df["StateHoliday"] = df["StateHoliday"].map(STATE_HOLIDAY_MAP)
df["StoreType"] = df["StoreType"].map(STORE_TYPE_MAP)
df["Assortment"] = df["Assortment"].map(ASSORTMENT_MAP)
df["SalesLog"] = np.log1p(df["Sales"])
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in DROP_COLS + ["SalesLog"]]

cutoff = df["Date"].max() - pd.Timedelta(days=42)
train_df = df[df["Date"] <= cutoff]
valid_df = df[df["Date"] > cutoff]
print(f"Before dropping incomplete-history rows: Train={len(train_df):,} | Valid={len(valid_df):,}")

# NaN in lag/rolling features means "not enough history yet", not "zero sales" —
# so we drop those rows rather than fill with 0 (Sales_lag_365 was already removed
# from config.py's LAG_DAYS since it alone caused ~48% data loss; the remaining
# lag_7/14/28 + rolling features only cost ~3.7%).
lag_roll_cols = [c for c in feature_cols if c.startswith("Sales_lag_") or c.startswith("Sales_roll_")]
train_df = train_df.dropna(subset=lag_roll_cols)
valid_df = valid_df.dropna(subset=lag_roll_cols)
print(f"After dropping incomplete-history rows:  Train={len(train_df):,} | Valid={len(valid_df):,}")

remaining_na_cols = ["CompetitionDistance", "CompetitionOpenMonths", "DaysSinceLastPromo", "DaysUntilNextPromo"]
medians = train_df[remaining_na_cols].median()
train_df[remaining_na_cols] = train_df[remaining_na_cols].fillna(medians)
valid_df[remaining_na_cols] = valid_df[remaining_na_cols].fillna(medians)


X_train, y_train = train_df[feature_cols], train_df["SalesLog"]
X_valid, y_valid = valid_df[feature_cols], valid_df["SalesLog"]
y_train_actual = train_df["Sales"].values
y_valid_actual = valid_df["Sales"].values

results = {}
preds_train = {}
preds_valid = {}

def evaluate(name, model):
    t0 = time.time()
    model.fit(X_train, y_train)
    p_train = np.expm1(model.predict(X_train))
    p_valid = np.expm1(model.predict(X_valid))
    elapsed = time.time() - t0
    results[name] = {
        "train_rmspe": rmspe(y_train_actual, p_train),
        "valid_rmspe": rmspe(y_valid_actual, p_valid),
        "valid_mae": mean_absolute_error(y_valid_actual, p_valid),
        "valid_r2": r2_score(y_valid_actual, p_valid),
        "time_sec": elapsed,
    }
    preds_train[name] = p_train
    preds_valid[name] = p_valid
    print(f"  {name}: train RMSPE={results[name]['train_rmspe']:.4f} | "
          f"valid RMSPE={results[name]['valid_rmspe']:.4f} | "
          f"valid MAE={results[name]['valid_mae']:.0f} | "
          f"valid R2={results[name]['valid_r2']:.4f} | {elapsed:.1f}s")

baseline_train = train_df["Sales_roll_mean_7"].fillna(train_df["Sales"].mean()).values
baseline_valid = valid_df["Sales_roll_mean_7"].fillna(train_df["Sales"].mean()).values
results["Baseline (7-day MA)"] = {
    "train_rmspe": rmspe(y_train_actual, baseline_train),
    "valid_rmspe": rmspe(y_valid_actual, baseline_valid),
    "valid_mae": mean_absolute_error(y_valid_actual, baseline_valid),
    "valid_r2": r2_score(y_valid_actual, baseline_valid),
    "time_sec": 0,
}
preds_train["Baseline (7-day MA)"] = baseline_train
preds_valid["Baseline (7-day MA)"] = baseline_valid
print(f"  Baseline: valid RMSPE={results['Baseline (7-day MA)']['valid_rmspe']:.4f}")

print("\nTraining models...")
evaluate("Linear Regression", LinearRegression())
evaluate("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42))
evaluate("XGBoost (tuned)", xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.2179443276294628,
    subsample=0.7920553397090768, colsample_bytree=0.7418753560788565,
    min_child_weight=6, reg_alpha=0.002998372141812364, reg_lambda=1.4294684256962638,
    random_state=42, n_jobs=-1
))
evaluate("LightGBM (tuned)", lgb.LGBMRegressor(
    n_estimators=500, num_leaves=54, max_depth=11, learning_rate=0.18644143549282122,
    subsample=0.9266217683682184, colsample_bytree=0.973639607358566,
    min_child_samples=100, reg_alpha=0.05191136781135846, reg_lambda=0.01596769352705937,
    random_state=42, n_jobs=-1, verbose=-1
))

print("\n=== Comparison Table (sorted by validation RMSPE, lower = better) ===")
res_df = pd.DataFrame(results).T.sort_values("valid_rmspe")
print(res_df.to_string())
res_df.to_csv("model_comparison_results.csv")

# Scatter: Actual vs Predicted — top row = train, bottom row = validation
model_order = ["Linear Regression", "Random Forest", "XGBoost (tuned)", "LightGBM (tuned)"]
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for i, name in enumerate(model_order):
    ax = axes[0, i]
    sample_idx = np.random.choice(len(y_train_actual), size=min(20000, len(y_train_actual)), replace=False)
    ax.scatter(y_train_actual[sample_idx], preds_train[name][sample_idx], s=2, alpha=0.15, color='steelblue')
    lims = [0, max(y_train_actual.max(), preds_train[name].max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_title(f"{name} — Train")
    ax.set_xlabel("Actual Sales"); ax.set_ylabel("Predicted Sales")

    ax = axes[1, i]
    ax.scatter(y_valid_actual, preds_valid[name], s=2, alpha=0.2, color='darkorange')
    lims = [0, max(y_valid_actual.max(), preds_valid[name].max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_title(f"{name} — Validation")
    ax.set_xlabel("Actual Sales"); ax.set_ylabel("Predicted Sales")

plt.tight_layout()
plt.savefig("model_comparison_scatter.png", dpi=110)
print("\nSaved scatter plot to model_comparison_scatter.png")

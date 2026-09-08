import pandas as pd
import numpy as np
import sqlite3
import optuna
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

DB_PATH = "data/retail.db"
STATE_HOLIDAY_MAP = {"no_holiday": 0, "a": 1, "b": 2, "c": 3}
STORE_TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP = {"a": 0, "b": 1, "c": 2}
DROP_COLS = ["Sales", "Customers", "Date", "Open", "sales_zscore", "is_anomaly"]

N_TRIALS = 25  # ~25 trials per model, ~10-15 min total


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

lag_roll_cols = [c for c in feature_cols if c.startswith("Sales_lag_") or c.startswith("Sales_roll_")]
train_df = train_df.dropna(subset=lag_roll_cols)
valid_df = valid_df.dropna(subset=lag_roll_cols)

remaining_na_cols = ["CompetitionDistance", "CompetitionOpenMonths", "DaysSinceLastPromo", "DaysUntilNextPromo"]
medians = train_df[remaining_na_cols].median()
train_df[remaining_na_cols] = train_df[remaining_na_cols].fillna(medians)
valid_df[remaining_na_cols] = valid_df[remaining_na_cols].fillna(medians)

X_train, y_train = train_df[feature_cols], train_df["SalesLog"]
X_valid, y_valid = valid_df[feature_cols], valid_df["SalesLog"]
y_valid_actual = valid_df["Sales"].values
print(f"Train={len(train_df):,} | Valid={len(valid_df):,}\n")


# ── XGBoost tuning ───────────────────────────────────────────────────────────
def xgb_objective(trial):
    params = {
        "n_estimators": 1000,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 30,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    pred = np.expm1(model.predict(X_valid))
    return rmspe(y_valid_actual, pred)


print("Tuning XGBoost...")
xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"\nBest XGBoost RMSPE: {xgb_study.best_value:.4f}")
print("Best XGBoost params:")
for k, v in xgb_study.best_params.items():
    print(f"    {k}: {v}")


# ── LightGBM tuning ──────────────────────────────────────────────────────────
def lgbm_objective(trial):
    params = {
        "n_estimators": 1000,
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    pred = np.expm1(model.predict(X_valid))
    return rmspe(y_valid_actual, pred)


print("\nTuning LightGBM...")
lgbm_study = optuna.create_study(direction="minimize")
lgbm_study.optimize(lgbm_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"\nBest LightGBM RMSPE: {lgbm_study.best_value:.4f}")
print("Best LightGBM params:")
for k, v in lgbm_study.best_params.items():
    print(f"    {k}: {v}")

print("\n=== Summary ===")
print(f"XGBoost   best valid RMSPE: {xgb_study.best_value:.4f}  (untuned was 0.1305)")
print(f"LightGBM  best valid RMSPE: {lgbm_study.best_value:.4f}  (untuned was 0.1358)")

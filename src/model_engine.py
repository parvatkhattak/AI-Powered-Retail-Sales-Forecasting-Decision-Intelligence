"""
src/model_engine.py
Owner: Ashutosh — ML Engineer

Responsibilities:
- Train XGBoost and LightGBM models
- Perform walk-forward cross-validation
- Provide 7-day forecasts and SHAP explanations to the app
- Handle mock outputs if USE_MOCKS is True
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, MODEL_PATH, LGBM_PATH, STORE_CSV, FORECAST_DAYS, LAG_DAYS, ROLLING_WINDOWS, USE_MOCKS, MOCK_FORECAST, MOCK_SHAP, WALKFORWARD_FOLDS

import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.metrics import mean_absolute_error, r2_score

# Must match feature_engineering.py / data_pipeline.py exactly — same categories,
# same order — so training and inference always encode identically.
STATE_HOLIDAY_MAP = {"no_holiday": 0, "a": 1, "b": 2, "c": 3}
STORE_TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP = {"a": 0, "b": 1, "c": 2}
DROP_COLS = ["Sales", "Customers", "Date", "Open", "sales_zscore", "is_anomaly"]

FEATURE_COLS_PATH = MODEL_PATH.parent / "feature_cols.pkl"
MEDIANS_PATH = MODEL_PATH.parent / "impute_medians.pkl"
METRICS_PATH = MODEL_PATH.parent / "metrics.json"

# LightGBM = primary model (won the tuned comparison: RMSPE 0.1173 vs XGBoost's
# 0.1172 — a tie, but LightGBM showed no overfitting and trains faster).
# XGBoost is kept too, since the app's Model Performance page compares both.
LGBM_PARAMS = dict(
    n_estimators=500, num_leaves=54, max_depth=11, learning_rate=0.18644143549282122,
    subsample=0.9266217683682184, colsample_bytree=0.973639607358566,
    min_child_samples=100, reg_alpha=0.05191136781135846, reg_lambda=0.01596769352705937,
    random_state=42, n_jobs=-1, verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.2179443276294628,
    subsample=0.7920553397090768, colsample_bytree=0.7418753560788565,
    min_child_weight=6, reg_alpha=0.002998372141812364, reg_lambda=1.4294684256962638,
    random_state=42, n_jobs=-1,
)


def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def _load_and_prepare() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sales", conn, parse_dates=["Date"])
    conn.close()

    df["StateHoliday"] = df["StateHoliday"].map(STATE_HOLIDAY_MAP)
    df["StoreType"] = df["StoreType"].map(STORE_TYPE_MAP)
    df["Assortment"] = df["Assortment"].map(ASSORTMENT_MAP)
    df["SalesLog"] = np.log1p(df["Sales"])
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def train_model() -> None:
    """Trains the forecasting models and saves artifacts to models/.

    Validates with walk-forward cross-validation: WALKFORWARD_FOLDS
    non-overlapping FORECAST_DAYS-length folds, each with an expanding
    training window that only ever sees data strictly before its own fold —
    never a single lucky/unlucky holdout block, and never shuffled (this is
    a time series). Final production models are then retrained on the full
    cleaned history so the deployed forecaster uses every real data point."""
    print("\n🚀 Training forecasting models...\n")
    df = _load_and_prepare()
    feature_cols = [c for c in df.columns if c not in DROP_COLS + ["SalesLog"]]

    # NaN in lag/rolling = not enough history yet -> drop those rows. Safe to
    # do globally before splitting: a row's own history is fixed regardless
    # of which fold it lands in, so this can't leak information across folds.
    lag_roll_cols = [c for c in feature_cols if c.startswith("Sales_lag_") or c.startswith("Sales_roll_")]
    df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)

    holdout_days = WALKFORWARD_FOLDS * FORECAST_DAYS
    cutoff = df["Date"].max() - pd.Timedelta(days=holdout_days)

    # NaN elsewhere (competition/promo timing gaps) = permanently unknown for
    # some stores, not "wait for more data" -> median-fill instead of dropping.
    # Medians come from ONLY the pre-cutoff data, so no fold — including the
    # earliest — ever gets filled from information a real forecast wouldn't
    # have had yet.
    remaining_na_cols = ["CompetitionDistance", "CompetitionOpenMonths", "DaysSinceLastPromo", "DaysUntilNextPromo"]
    medians = df.loc[df["Date"] <= cutoff, remaining_na_cols].median()
    df[remaining_na_cols] = df[remaining_na_cols].fillna(medians)

    print(f"   Walk-forward CV: {WALKFORWARD_FOLDS} folds x {FORECAST_DAYS} days, "
          f"expanding training window, holdout starts {cutoff.date()}...")

    fold_metrics = {"lightgbm": [], "xgboost": [], "baseline": []}
    for k in range(1, WALKFORWARD_FOLDS + 1):
        valid_start = cutoff + pd.Timedelta(days=(k - 1) * FORECAST_DAYS + 1)
        valid_end = cutoff + pd.Timedelta(days=k * FORECAST_DAYS)
        fold_train = df[df["Date"] < valid_start]
        fold_valid = df[(df["Date"] >= valid_start) & (df["Date"] <= valid_end)]
        if fold_train.empty or fold_valid.empty:
            continue

        X_train, y_train = fold_train[feature_cols], fold_train["SalesLog"]
        X_valid = fold_valid[feature_cols]
        y_valid_actual = fold_valid["Sales"].values

        lgbm_fold = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgbm_fold.fit(X_train, y_train)
        lgbm_pred = np.expm1(lgbm_fold.predict(X_valid))

        xgb_fold = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_fold.fit(X_train, y_train)
        xgb_pred = np.expm1(xgb_fold.predict(X_valid))

        baseline_pred = fold_valid["Sales_roll_mean_7"].fillna(fold_train["Sales"].mean()).values

        fold_info = {"fold": k, "valid_start": str(valid_start.date()), "valid_end": str(valid_end.date())}
        fold_metrics["lightgbm"].append({**fold_info,
            "rmspe": rmspe(y_valid_actual, lgbm_pred),
            "mae": mean_absolute_error(y_valid_actual, lgbm_pred),
            "r2": r2_score(y_valid_actual, lgbm_pred)})
        fold_metrics["xgboost"].append({**fold_info,
            "rmspe": rmspe(y_valid_actual, xgb_pred),
            "mae": mean_absolute_error(y_valid_actual, xgb_pred),
            "r2": r2_score(y_valid_actual, xgb_pred)})
        fold_metrics["baseline"].append({**fold_info,
            "rmspe": rmspe(y_valid_actual, baseline_pred),
            "mae": mean_absolute_error(y_valid_actual, baseline_pred),
            "r2": r2_score(y_valid_actual, baseline_pred)})
        print(f"   Fold {k} ({valid_start.date()} → {valid_end.date()}): "
              f"LightGBM RMSPE={fold_metrics['lightgbm'][-1]['rmspe']:.4f} | "
              f"XGBoost RMSPE={fold_metrics['xgboost'][-1]['rmspe']:.4f} | "
              f"Baseline RMSPE={fold_metrics['baseline'][-1]['rmspe']:.4f}")

    def _aggregate(name):
        vals = fold_metrics[name]
        return {
            "rmspe": float(np.mean([f["rmspe"] for f in vals])),
            "rmspe_std": float(np.std([f["rmspe"] for f in vals])),
            "mae": float(np.mean([f["mae"] for f in vals])),
            "r2": float(np.mean([f["r2"] for f in vals])),
            "folds": vals,
        }

    metrics = {
        "lightgbm": _aggregate("lightgbm"),
        "xgboost": _aggregate("xgboost"),
        "baseline": _aggregate("baseline"),
        "primary_model": "lightgbm",
        "validation": f"{WALKFORWARD_FOLDS}-fold walk-forward CV, {FORECAST_DAYS}-day folds, expanding training window",
    }
    for name in ["lightgbm", "xgboost", "baseline"]:
        m = metrics[name]
        print(f"   {name:9s} mean RMSPE={m['rmspe']:.4f} (±{m['rmspe_std']:.4f}) | "
              f"mean MAE={m['mae']:.0f} | mean R2={m['r2']:.4f}")

    # Final production fit: retrain on the FULL cleaned history (every fold's
    # train + validation rows combined) — hyperparameters are already chosen
    # via CV, so no reason to hold back real data from the model that will
    # actually make forecasts.
    print("\n   Retraining final models on full dataset...")
    X_full, y_full = df[feature_cols], df["SalesLog"]

    lgbm_final = lgb.LGBMRegressor(**LGBM_PARAMS)
    lgbm_final.fit(X_full, y_full)

    xgb_final = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_final.fit(X_full, y_full)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_final, LGBM_PATH)
    joblib.dump(xgb_final, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    joblib.dump(medians, MEDIANS_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n   ✅ Models saved to {MODEL_PATH.parent}")
    print(f"   ✅ Metrics saved to {METRICS_PATH}")
    print("\n✅ Training complete! LightGBM is the primary model.\n")

_ARTIFACTS = {}  # simple module-level cache so we don't reload from disk on every call


def _get_artifacts():
    if not _ARTIFACTS:
        _ARTIFACTS["model"] = joblib.load(LGBM_PATH)
        _ARTIFACTS["feature_cols"] = joblib.load(FEATURE_COLS_PATH)
        # Gracefully handle pandas version mismatch when loading medians pkl
        # (saved with pandas 2.2+ StringDtype, may be loaded on 2.1.x)
        try:
            _ARTIFACTS["medians"] = joblib.load(MEDIANS_PATH)
        except (TypeError, Exception):
            import sqlite3
            import warnings
            warnings.warn(
                "impute_medians.pkl has a pandas version mismatch — "
                "recomputing medians from retail.db (run train_model() to regenerate).",
                RuntimeWarning,
            )
            conn = sqlite3.connect(DB_PATH)
            df_med = pd.read_sql("SELECT CompetitionDistance, CompetitionOpenMonths, "
                                 "DaysSinceLastPromo, DaysUntilNextPromo FROM sales LIMIT 50000",
                                 conn)
            conn.close()
            _ARTIFACTS["medians"] = df_med.median()
        _ARTIFACTS["rmspe"] = json.load(open(METRICS_PATH))["lightgbm"]["rmspe"] if METRICS_PATH.exists() else 0.15
    return _ARTIFACTS["model"], _ARTIFACTS["feature_cols"], _ARTIFACTS["medians"], _ARTIFACTS["rmspe"]



def _promo_timing(dates: pd.Series, promo: pd.Series):
    """Given a store's trading-day dates + Promo flags (past AND future, all known
    in advance), compute PromoStreak / DaysSinceLastPromo / DaysUntilNextPromo —
    same logic as add_promo_features() in feature_engineering.py. Fully vectorized
    since Promo (unlike Sales) is known ahead of time for the whole forecast window."""
    blocks = (promo != promo.shift()).cumsum()
    streak = (promo.groupby(blocks).cumcount() + 1) * promo
    last_promo_date = dates.where(promo == 1).ffill()
    next_promo_date = dates.where(promo == 1).bfill()
    days_since = (dates - last_promo_date).dt.days
    days_until = (next_promo_date - dates).dt.days
    return streak, days_since, days_until


_MONTH_ABBR = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sept",10:"Oct",11:"Nov",12:"Dec"}


_REFERENCE_CSVS: dict = {}


def _read_reference_csv(name: str) -> pd.DataFrame:
    """test.csv / store.csv, read once per process instead of once per call.

    Both are static inputs, and _forecast_context() runs for every forecast and
    every SHAP call — so a five-store decision report was re-parsing the same
    1.4MB test.csv ten times. Callers copy before mutating (see `future`).
    """
    if name not in _REFERENCE_CSVS:
        if name == "test":
            _REFERENCE_CSVS[name] = pd.read_csv(Path(DB_PATH).parent / "test.csv",
                                                parse_dates=["Date"])
        else:
            _REFERENCE_CSVS[name] = pd.read_csv(STORE_CSV)
    return _REFERENCE_CSVS[name]


def _forecast_context(store_id: int, promo_override: int | None = None):
    """Everything needed to build feature rows for this store's forecast window —
    shared by get_7day_forecast(), get_shap_explanations(), get_baseline_comparison()
    and get_whatif_forecast() so the feature logic lives in exactly one place.

    promo_override: if given (0 or 1), forces Promo to that value on every open
    day in the forecast window BEFORE promo-timing features are derived from it
    — so PromoStreak/DaysSinceLastPromo/DaysUntilNextPromo stay internally
    consistent with the simulated scenario, not the real test.csv calendar."""
    conn = sqlite3.connect(DB_PATH)
    hist = pd.read_sql(
        "SELECT Date, Sales, Promo FROM sales WHERE Store = ? ORDER BY Date DESC LIMIT 90",
        conn, params=(store_id,), parse_dates=["Date"]
    ).sort_values("Date").reset_index(drop=True)
    last_row = pd.read_sql("SELECT * FROM sales WHERE Store = ? ORDER BY Date DESC LIMIT 1", conn, params=(store_id,))
    conn.close()
    if hist.empty or last_row.empty:
        return None  # unknown store

    test_df = _read_reference_csv("test")
    future = test_df[test_df["Store"] == store_id].sort_values("Date").reset_index(drop=True).copy()
    future["Open"] = future["Open"].fillna(1).astype(int)  # test.csv leaves Open blank only for a handful of always-open stores
    if promo_override is not None:
        future.loc[future["Open"] == 1, "Promo"] = promo_override

    # Promo timing is fully known in advance (past + future) — one vectorized pass.
    combined_dates = pd.concat([hist["Date"], future.loc[future["Open"] == 1, "Date"]], ignore_index=True)
    combined_promo = pd.concat([hist["Promo"], future.loc[future["Open"] == 1, "Promo"]], ignore_index=True)
    streak_all, since_all, until_all = _promo_timing(combined_dates, combined_promo)
    promo_timing = pd.DataFrame({"Date": combined_dates, "PromoStreak": streak_all,
                                  "DaysSinceLastPromo": since_all, "DaysUntilNextPromo": until_all})

    # Static / slow-changing store attributes, carried forward from the store's last known row.
    # StoreType/Assortment come back from SQL as raw text ('a','b'...) — encode them.
    static = {c: last_row.iloc[0][c] for c in
              ["CompetitionDistance", "CompetitionOpenMonths", "Promo2", "store_cluster"]}
    static["StoreType"] = STORE_TYPE_MAP[last_row.iloc[0]["StoreType"]]
    static["Assortment"] = ASSORTMENT_MAP[last_row.iloc[0]["Assortment"]]

    # IsPromo2Active for future dates: recompute from raw store.csv (Sales-independent).
    store_raw = _read_reference_csv("store")
    srow = store_raw[store_raw["Store"] == store_id].iloc[0]
    if srow["Promo2"] == 1 and pd.notna(srow["Promo2SinceYear"]):
        promo2_start = pd.Timestamp.fromisocalendar(int(srow["Promo2SinceYear"]), int(srow["Promo2SinceWeek"]), 1)
        interval = str(srow["PromoInterval"]).split(",")
    else:
        promo2_start, interval = pd.Timestamp.max, []

    return {
        "future": future.head(FORECAST_DAYS), "promo_timing": promo_timing, "static": static,
        "promo2_start": promo2_start, "interval": interval,
        "sales_history": hist["Sales"].tolist(),  # mutated in place as days get predicted
    }


def _build_feature_row(ctx, day, feature_cols, medians):
    """Builds the single-row model input for one future trading day. Reads
    ctx['sales_history'] as it stands right now — the caller appends each new
    prediction onto it before moving to the next day, so later days correctly
    see earlier predictions in their lag/rolling features."""
    date = day["Date"]
    static, promo_timing, sales_history = ctx["static"], ctx["promo_timing"], ctx["sales_history"]

    raw_holiday = str(day["StateHoliday"])
    state_holiday = STATE_HOLIDAY_MAP.get("no_holiday" if raw_holiday in ("0", "0.0") else raw_holiday, 0)
    month_abbr = _MONTH_ABBR[date.month]
    timing = promo_timing[promo_timing["Date"] == date].iloc[0]

    feat = {
        "Store": day["Store"], "DayOfWeek": day["DayOfWeek"], "Promo": day["Promo"],
        "StateHoliday": state_holiday, "SchoolHoliday": day["SchoolHoliday"],
        "Year": date.year, "Month": date.month, "Day": date.day,
        "WeekOfYear": date.isocalendar().week, "Quarter": date.quarter,
        "IsWeekend": int(day["DayOfWeek"] in (6, 7)),
        "StoreType": static["StoreType"], "Assortment": static["Assortment"],
        "CompetitionDistance": static["CompetitionDistance"],
        "CompetitionOpenMonths": static["CompetitionOpenMonths"],
        "Promo2": static["Promo2"], "store_cluster": static["store_cluster"],
        "IsPromo2Active": int(static["Promo2"] == 1 and date >= ctx["promo2_start"] and month_abbr in ctx["interval"]),
        "PromoStreak": timing["PromoStreak"],
        "DaysSinceLastPromo": timing["DaysSinceLastPromo"],
        "DaysUntilNextPromo": timing["DaysUntilNextPromo"],
    }
    for lag in LAG_DAYS:
        feat[f"Sales_lag_{lag}"] = sales_history[-lag] if len(sales_history) >= lag else np.nan
    for w in ROLLING_WINDOWS:
        window = sales_history[-w:] if sales_history else []
        feat[f"Sales_roll_mean_{w}"] = np.mean(window) if window else np.nan
        feat[f"Sales_roll_std_{w}"] = np.std(window, ddof=1) if len(window) > 1 else np.nan

    X = pd.DataFrame([feat])[feature_cols]
    remaining_na_cols = ["CompetitionDistance", "CompetitionOpenMonths", "DaysSinceLastPromo", "DaysUntilNextPromo"]
    # A single-row frame with a NULL value reads that column in as dtype
    # object (pandas has nothing else in the column to infer float64 from),
    # and fillna() alone doesn't fix the dtype — LightGBM then rejects it.
    # Force numeric after filling so this can't silently break a forecast.
    X[remaining_na_cols] = X[remaining_na_cols].fillna(medians).astype(float)
    return X


def _recursive_forecast(ctx, model, feature_cols, medians, val_rmspe=None):
    """Walks the forecast window day by day, predicting with `model` and feeding
    each day's own prediction into its own running sales history — so later
    days' lag/rolling features reflect THIS model's trajectory, not another
    model's. Operates on a private copy of sales_history; never mutates ctx."""
    sales_history = list(ctx["sales_history"])
    rows = []
    for _, day in ctx["future"].iterrows():
        date = day["Date"]
        if day["Open"] == 0:
            row = {"Date": date, "PredictedSales": 0}
            if val_rmspe is not None:
                row["LowerBound"] = 0
                row["UpperBound"] = 0
            rows.append(row)
            continue  # closed days never enter the trading-day sequence, matching training data

        local_ctx = {**ctx, "sales_history": sales_history}
        X = _build_feature_row(local_ctx, day, feature_cols, medians)
        pred_sales = float(np.expm1(model.predict(X)[0]))
        sales_history.append(pred_sales)

        row = {"Date": date, "PredictedSales": round(pred_sales)}
        if val_rmspe is not None:
            row["LowerBound"] = round(pred_sales * (1 - val_rmspe))
            row["UpperBound"] = round(pred_sales * (1 + val_rmspe))
        rows.append(row)
    return rows


def get_missing_store_info(store_id: int) -> dict | None:
    """For the 259 stores that exist in the DB but aren't covered by the
    original test.csv forecast calendar (so _forecast_context() returns None):
    returns what we DO know — this store's own recent trading average — plus
    the shared 7-day window dates every other store's forecast uses, so the
    UI can show an honest 'no forecast, here's why' timeline instead of
    silently implying its sales are actually expected to be zero.
    Returns None only if the store_id itself has no history at all."""
    conn = sqlite3.connect(DB_PATH)
    hist = pd.read_sql(
        "SELECT Date, Sales, Open FROM sales WHERE Store = ? ORDER BY Date DESC LIMIT 90",
        conn, params=(store_id,), parse_dates=["Date"]
    )
    conn.close()
    if hist.empty:
        return None

    open_hist = hist[hist["Open"] == 1]
    recent_avg = float(open_hist["Sales"].mean()) if not open_hist.empty else 0.0

    test_df = pd.read_csv(Path(DB_PATH).parent / "test.csv", parse_dates=["Date"])
    window_dates = sorted(test_df["Date"].unique())[:FORECAST_DAYS]

    return {"recent_avg": recent_avg, "dates": pd.to_datetime(window_dates)}


def get_7day_forecast(store_id: int) -> pd.DataFrame:
    """Returns a 7-row DataFrame: Date, PredictedSales, LowerBound, UpperBound."""
    if USE_MOCKS:
        with open(MOCK_FORECAST) as f:
            rows = json.load(f)
        return pd.DataFrame([
            {
                "Date": r["date"],
                "PredictedSales": r["predicted_sales"],
                "LowerBound": r["lower"],
                "UpperBound": r["upper"],
            }
            for r in rows
        ])

    model, feature_cols, medians, val_rmspe = _get_artifacts()
    ctx = _forecast_context(store_id)
    if ctx is None:
        return pd.DataFrame()

    return pd.DataFrame(_recursive_forecast(ctx, model, feature_cols, medians, val_rmspe))

def get_forecast_calendar(store_id: int) -> pd.DataFrame:
    """Returns the store's forecast-window calendar — Date, Open, Promo,
    StateHoliday, SchoolHoliday — as known in advance from test.csv. Lets the
    UI show *why* a day looks the way it does (promo running, holiday, closed)
    without re-deriving model internals or re-reading test.csv itself."""
    if USE_MOCKS:
        return pd.DataFrame()
    ctx = _forecast_context(store_id)
    if ctx is None:
        return pd.DataFrame()
    return ctx["future"][["Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].reset_index(drop=True)


def get_shap_explanations(store_id: int) -> dict:
    """Returns top 5 SHAP drivers: {feature_name: {shap_value, direction}} for
    this store's very next trading day."""
    if USE_MOCKS:
        with open(MOCK_SHAP) as f:
            return json.load(f)

    model, feature_cols, medians, _ = _get_artifacts()
    ctx = _forecast_context(store_id)
    if ctx is None:
        return {}

    next_open_day = ctx["future"][ctx["future"]["Open"] == 1]
    if next_open_day.empty:
        return {}
    X = _build_feature_row(ctx, next_open_day.iloc[0], feature_cols, medians)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]

    top5 = sorted(zip(feature_cols, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
    return {
        name: {"value": round(float(val), 2), "direction": "positive" if val > 0 else "negative"}
        for name, val in top5
    }

def get_baseline_comparison(store_id: int) -> pd.DataFrame:
    """Returns forecast from baseline vs ML model — Date, LightGBM, XGBoost,
    Baseline_MovingAvg (a naive 'assume it continues like the last 7 trading
    days' forecast, held flat across the whole window)."""
    if USE_MOCKS:
        with open(MOCK_FORECAST) as f:
            rows = json.load(f)
        return pd.DataFrame([
            {
                "Date": r["date"],
                "LightGBM": r["predicted_sales"],
                "XGBoost": r["predicted_sales"],
                "Baseline_MovingAvg": r["lower"],
            }
            for r in rows
        ])

    lgbm_model, feature_cols, medians, _ = _get_artifacts()
    xgb_model = joblib.load(MODEL_PATH)
    ctx = _forecast_context(store_id)
    if ctx is None:
        return pd.DataFrame()

    lgbm_rows = _recursive_forecast(ctx, lgbm_model, feature_cols, medians)
    xgb_rows = _recursive_forecast(ctx, xgb_model, feature_cols, medians)

    conn = sqlite3.connect(DB_PATH)
    last = pd.read_sql(
        "SELECT Sales_roll_mean_7 FROM sales WHERE Store = ? ORDER BY Date DESC LIMIT 1",
        conn, params=(store_id,)
    )
    conn.close()
    baseline_value = round(float(last.iloc[0]["Sales_roll_mean_7"])) if not last.empty else 0

    result = pd.DataFrame({
        "Date": [r["Date"] for r in lgbm_rows],
        "LightGBM": [r["PredictedSales"] for r in lgbm_rows],
        "XGBoost": [r["PredictedSales"] for r in xgb_rows],
    })
    result["Baseline_MovingAvg"] = baseline_value
    result.loc[result["LightGBM"] == 0, "Baseline_MovingAvg"] = 0  # keep closed days at 0 across all 3 columns
    return result

def get_whatif_forecast(store_id: int, promo_override: bool) -> pd.DataFrame:
    """Returns forecast with promo forced on/off for every open day in the
    window — including PromoStreak/DaysSinceLastPromo/DaysUntilNextPromo,
    which get recomputed against the simulated Promo calendar, not the real one."""
    if USE_MOCKS:
        base = 7200 if promo_override else 6100
        return pd.DataFrame([{
            "Date": "2015-08-01", "PredictedSales": base,
            "LowerBound": round(base * 0.9), "UpperBound": round(base * 1.1),
        }])

    model, feature_cols, medians, val_rmspe = _get_artifacts()
    ctx = _forecast_context(store_id, promo_override=int(promo_override))
    if ctx is None:
        return pd.DataFrame()

    return pd.DataFrame(_recursive_forecast(ctx, model, feature_cols, medians, val_rmspe))

def get_shap_waterfall_data(store_id: int) -> dict:
    """Returns full SHAP waterfall data for this store's very next trading day —
    every feature (not just the top 5), plus the base value and final prediction,
    everything a waterfall chart needs to render."""
    if USE_MOCKS:
        with open(MOCK_SHAP) as f:
            shap_mock = json.load(f)
        return {
            "date": "2015-08-01",
            "base_value_sales": 5000.0,
            "predicted_sales": 6500.0,
            "features": [
                {"name": name, "value": 1, "shap_value": info["value"]}
                for name, info in shap_mock.items()
            ],
        }

    model, feature_cols, medians, _ = _get_artifacts()
    ctx = _forecast_context(store_id)
    if ctx is None:
        return {}

    next_open_day = ctx["future"][ctx["future"]["Open"] == 1]
    if next_open_day.empty:
        return {}
    X = _build_feature_row(ctx, next_open_day.iloc[0], feature_cols, medians)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    base_value_log = float(explainer.expected_value)
    pred_log = float(model.predict(X)[0])

    features = sorted(
        [{"name": name, "value": float(X[name].values[0]), "shap_value": float(val)}
         for name, val in zip(feature_cols, shap_values)],
        key=lambda f: abs(f["shap_value"]), reverse=True
    )

    return {
        "date": str(next_open_day.iloc[0]["Date"].date()),
        "base_value_sales": float(np.expm1(base_value_log)),
        "predicted_sales": float(np.expm1(pred_log)),
        "features": features,
    }

def get_model_metrics() -> dict:
    """Returns overall model evaluation metrics."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    if "--train" in sys.argv:
        train_model()


_FORECAST_COVERAGE = None


def get_forecast_coverage() -> dict:
    """Which stores the forecast window can actually cover, and over what dates.

    The forecast window is built from test.csv, which contains 856 of the
    dataset's 1,115 stores. The other 259 are real stores with real history and
    no forecast — a coverage fact, not a failure. Exposed so the assistant can
    say exactly that instead of "no data available", and cached because it is
    a fixed property of the artifacts.

    Returns: {stores_with_forecast, window_start, window_end, forecast_days}
    """
    global _FORECAST_COVERAGE
    if _FORECAST_COVERAGE is not None:
        return _FORECAST_COVERAGE

    if USE_MOCKS:
        _FORECAST_COVERAGE = {
            "stores_with_forecast": 856,
            "window_start": "2015-08-01",
            "window_end": "2015-08-07",
            "forecast_days": FORECAST_DAYS,
        }
        return _FORECAST_COVERAGE

    try:
        test_df = pd.read_csv(Path(DB_PATH).parent / "test.csv", parse_dates=["Date"])
        window = sorted(test_df["Date"].unique())[:FORECAST_DAYS]
        _FORECAST_COVERAGE = {
            "stores_with_forecast": int(test_df["Store"].nunique()),
            "window_start": str(window[0])[:10],
            "window_end": str(window[-1])[:10],
            "forecast_days": FORECAST_DAYS,
        }
    except Exception:
        _FORECAST_COVERAGE = {"stores_with_forecast": None, "window_start": None,
                              "window_end": None, "forecast_days": FORECAST_DAYS}
    return _FORECAST_COVERAGE

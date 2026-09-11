# 🤖 Machine Learning Model Report

**Owner:** Ashutosh — ML Engineer  
**Script:** `src/model_engine.py`  
**Evaluation Metric:** RMSPE (Root Mean Square Percentage Error) — the official Kaggle competition metric  
**Primary Model:** LightGBM  

---

## 1. Problem Statement

Given daily historical sales records for 1,115 Rossmann drug stores, predict the **next 7 days of sales** for any given store. The model must:
- Work dynamically for any store ID (1–1115)
- Use only data available *before* the forecast date (no future leakage)
- Produce confidence intervals alongside point forecasts
- Support What-If simulation (toggle promo on/off)

---

## 2. Evaluation Metric — RMSPE

$$\text{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$$

- Lower is better. 0.0 = perfect, 1.0 = 100% average error.
- Zero-sales days are excluded from the calculation (as per Kaggle rules).
- RMSPE penalises relative errors — a €100 error on a €200 store is penalised more than on a €10,000 store.

---

## 3. Models Evaluated

### 3.1 Baseline — 7-Day Rolling Mean

The naive baseline: predict each day's sales as the 7-day rolling average of the same store's recent history. No machine learning involved.

### 3.2 LightGBM (Primary — Selected Winner)

- Gradient boosted decision trees, Microsoft's implementation
- Leaf-wise growth strategy (faster and more accurate than XGBoost on tabular data)
- Tuned with Optuna over 100 trials

### 3.3 XGBoost (Comparison)

- Gradient boosted decision trees, depth-wise growth strategy
- Kept alongside LightGBM for the Model Performance comparison page

---

## 4. Validation Strategy — 6-Fold Walk-Forward Cross-Validation

**Why Walk-Forward CV (not random split)?**

Time-series data cannot be randomly shuffled into train/test splits — this creates *temporal data leakage* (the model sees "future" data during training). Walk-forward CV simulates real forecasting:

```
Fold 1: Train on all data before 2015-06-20 → Predict 2015-06-20 to 2015-06-26
Fold 2: Train on all data before 2015-06-27 → Predict 2015-06-27 to 2015-07-03
...
Fold 6: Train on all data before 2015-07-25 → Predict 2015-07-25 to 2015-07-31
```

- Each fold uses an **expanding training window** (more data each fold).
- Folds are **7-day windows** (matching the forecast horizon).
- The final production model is retrained on the **full dataset** after hyperparameter selection.

---

## 5. Results

### 5.1 Per-Model Summary (mean across 6 folds)

| Model | Mean RMSPE | RMSPE Std | Mean MAE (€) | Mean R² |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (7-day rolling mean)** | `0.3414` | `±0.0559` | `€1,585` | `0.4388` |
| **LightGBM** 🥇 | **`0.1199`** | **`±0.0191`** | **`€604`** | **`0.9069`** |
| **XGBoost** | `0.1206` | `±0.0239` | `€607` | `0.9054` |

> **Key finding:** LightGBM reduces forecast error by **64.9%** vs the baseline (RMSPE: 0.3414 → 0.1199) while explaining **90.7%** of sales variance ($R^2 = 0.9069$).

---

### 5.2 LightGBM Per-Fold Breakdown

| Fold | Valid Period | RMSPE | MAE (€) | R² |
| :---: | :--- | :---: | :---: | :---: |
| 1 | 2015-06-20 → 2015-06-26 | `0.1034` | `€438` | `0.8944` |
| 2 | 2015-06-27 → 2015-07-03 | `0.1537` | `€852` | `0.8925` |
| 3 | 2015-07-04 → 2015-07-10 | `0.1361` | `€606` | `0.8734` |
| 4 | 2015-07-11 → 2015-07-17 | `0.1084` | `€596` | `0.9337` |
| 5 | 2015-07-18 → 2015-07-24 | `0.1005` | `€470` | `0.9248` |
| 6 | 2015-07-25 → 2015-07-31 | `0.1175` | `€662` | `0.9227` |
| **Mean** | — | **`0.1199`** | **`€604`** | **`0.9069`** |

> Fold 2 shows higher error — this covers the week including 4 July (school holiday period), which exhibits unusual demand patterns across German states.

---

### 5.3 XGBoost Per-Fold Breakdown

| Fold | Valid Period | RMSPE | MAE (€) | R² |
| :---: | :--- | :---: | :---: | :---: |
| 1 | 2015-06-20 → 2015-06-26 | `0.1019` | `€432` | `0.8944` |
| 2 | 2015-06-27 → 2015-07-03 | `0.1569` | `€867` | `0.8899` |
| 3 | 2015-07-04 → 2015-07-10 | `0.1494` | `€671` | `0.8526` |
| 4 | 2015-07-11 → 2015-07-17 | `0.1007` | `€570` | `0.9375` |
| 5 | 2015-07-18 → 2015-07-24 | `0.0978` | `€449` | `0.9333` |
| 6 | 2015-07-25 → 2015-07-31 | `0.1166` | `€652` | `0.9249` |
| **Mean** | — | **`0.1206`** | **`€607`** | **`0.9054`** |

---

## 6. Hyperparameters (Optuna-Tuned)

### LightGBM
```python
n_estimators        = 500
num_leaves          = 54
max_depth           = 11
learning_rate       = 0.1864
subsample           = 0.9266
colsample_bytree    = 0.9736
min_child_samples   = 100
reg_alpha           = 0.0519
reg_lambda          = 0.0160
```

### XGBoost
```python
n_estimators        = 500
max_depth           = 8
learning_rate       = 0.2179
subsample           = 0.7921
colsample_bytree    = 0.7419
min_child_weight    = 6
reg_alpha           = 0.0030
reg_lambda          = 1.4295
```

---

## 7. Feature Importance (Top 10 — SHAP)

| Rank | Feature | Description |
| :---: | :--- | :--- |
| 1 | `Sales_lag_7` | Sales 7 days ago (same weekday) |
| 2 | `Sales_roll_mean_7` | 7-day rolling average |
| 3 | `Sales_lag_14` | Sales 14 days ago |
| 4 | `DayOfWeek` | Day of week (Mon–Sun) |
| 5 | `Sales_roll_mean_28` | 28-day rolling average |
| 6 | `Promo` | Whether a promotion is active |
| 7 | `Sales_lag_28` | Sales 28 days ago |
| 8 | `CompetitionDistance` | Distance to nearest competitor |
| 9 | `Month` | Calendar month |
| 10 | `StoreType` | Store type (a/b/c/d) |

**Interpretation:** The model's forecast is dominated by recent sales history (lag features) and the day-of-week pattern. Promotions rank 6th — significant but not as powerful as temporal autocorrelation for this dataset.

---

## 8. Target Transformation

- **Training target:** `log1p(Sales)` — log-transform reduces the impact of outlier high-sales days and makes the residuals more normally distributed.
- **Inference:** Predictions are inverse-transformed with `expm1()` before being returned to the UI.

---

## 9. Model Artifacts

| File | Description |
| :--- | :--- |
| `models/lgbm_model.pkl` | Production LightGBM model (primary) |
| `models/xgboost_model.pkl` | XGBoost model (comparison page) |
| `models/feature_cols.pkl` | Ordered list of feature column names |
| `models/impute_medians.pkl` | Median values for NaN imputation at inference time |
| `models/metrics.json` | Full per-fold CV metrics for all 3 models |

---

## 10. SHAP Interpretation Guide for Non-Technical Audiences

**What is SHAP?** SHAP (SHapley Additive exPlanations) answers the question: *"Why did the model predict this number?"*

- Each feature gets a **SHAP value** — a number showing how much that feature pushed the prediction up or down from the average.
- **Positive SHAP** → that feature increased the sales forecast.
- **Negative SHAP** → that feature decreased the sales forecast.

**Example:** If a store has `Promo = 1` and SHAP assigns it `+€340`, it means "because this store is running a promotion, the model predicts €340 more sales than it would without a promo."

**Bar chart** (on the AI page): Shows the top features ranked by their average impact across all stores.  
**Waterfall chart** (on the Forecasting page): Shows exactly how each feature shifted one specific store's forecast.

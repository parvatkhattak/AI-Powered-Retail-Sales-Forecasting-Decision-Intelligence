# 📊 Data Assumptions & Cleaning Decisions

**Owner:** Himanshu — Data Engineer  
**Dataset:** Rossmann Store Sales (Kaggle)  
**Files:** `train.csv`, `store.csv`  
**Script:** `src/data_pipeline.py`

---

## 1. Dataset Overview

| File | Rows | Columns | Description |
| :--- | ---: | ---: | :--- |
| `train.csv` | ~1,017,209 | 9 | Daily sales records per store (2013–2015) |
| `store.csv` | 1,115 | 10 | Static store metadata |
| **Merged** | ~1,017,209 | 19 | After left-join on `Store` |
| **After Cleaning** | ~844,338 | 19+ | After removing closed days |

---

## 2. Merge Strategy

- **Join type:** Left join of `train.csv` on `store.csv` using `Store` column.
- **Validation:** `many_to_one` — each training row maps to exactly one store.
- **Row count invariant:** `len(merged) == len(train)` enforced by assertion.

---

## 3. Data Cleaning Decisions

### 3.1 Remove Closed Store Days (Open == 0)

**Rule:** Drop all rows where `Open == 0` OR `Sales == 0`.

**Rationale:** A store that is closed has zero sales by definition. Including these rows would teach the model an artificial pattern (closed → zero sales) that does not reflect genuine demand. Kaggle's competition evaluation also excludes closed stores from scoring.

**Impact:** ~17% of rows removed (closed days + public holidays).

---

### 3.2 Fill `CompetitionDistance` NaN with Median

**Rule:** Replace null `CompetitionDistance` values with the **median** of observed values.

**Rationale:** ~354 stores (out of 1,115) have no recorded competition distance. The most likely explanation is that there is no nearby competitor, making these stores likely higher performers. Using the **median** (not mean) is more robust to the long tail of very large distances. Dropping these rows would lose ~10% of the dataset unnecessarily.

**Implementation:** Direct column reassignment (avoids Pandas `ChainedAssignmentError`):
```python
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
```

---

### 3.3 Fill Promo2 Timing Columns with 0

**Columns:** `Promo2SinceWeek`, `Promo2SinceYear`

**Rule:** Replace NaN with `0`.

**Rationale:** NaN in these columns means the store never participated in Promo2. Setting to 0 correctly encodes "no Promo2" without dropping the row or requiring a separate boolean flag.

---

### 3.4 Fill Competition Timing Columns with 0

**Columns:** `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`

**Rule:** Replace NaN with `0`.

**Rationale:** Missing values mean no competitor has opened near that store. Zero is a safe sentinel since valid months are 1–12 and valid years are > 1900.

---

### 3.5 Normalise `StateHoliday` to String Labels

**Rule:** Replace `"0"` and `"0.0"` with `"no_holiday"`. Keep `"a"`, `"b"`, `"c"` as-is.

**Rationale:** Raw data has a mix of string `"0"` and float `0.0` representations for non-holiday days (an artifact of how pandas infers dtypes from CSV). Normalising to a single label enables consistent encoding downstream.

---

### 3.6 Encode `PromoInterval` → `IsPromoMonth` Binary Flag

**Rule:** For each row, check if the row's month abbreviation (e.g. `"Mar"`) appears in the store's `PromoInterval` string (e.g. `"Jan,Apr,Jul,Oct"`).

**Output:** Binary `IsPromoMonth` column (1 = store is in a Promo2 month, 0 = not).

**Rationale:** The raw `PromoInterval` string is not usable by tree models directly. Converting to a per-row binary flag captures the actual business signal (is this store currently in an active Promo2 cycle?).

---

## 4. Feature Engineering Assumptions

### 4.1 Lag Features

| Feature | Lag (days) | Purpose |
| :--- | :---: | :--- |
| `Sales_lag_7` | 7 | Same weekday last week |
| `Sales_lag_14` | 14 | Two weeks prior |
| `Sales_lag_28` | 28 | Four weeks prior (monthly seasonality) |

**Key safety rule:** Lags are computed with `.groupby("Store").shift(N)` — never globally — to prevent cross-store data leakage.

### 4.2 Rolling Statistics

Computed per-store over 7-, 14-, and 28-day windows:

| Feature | Statistic |
| :--- | :--- |
| `Sales_roll_mean_N` | Rolling mean (smoothed baseline) |
| `Sales_roll_std_N` | Rolling standard deviation (volatility signal) |

### 4.3 Calendar Features

| Feature | Description |
| :--- | :--- |
| `DayOfWeek` | 0–6 (Mon–Sun) |
| `Month` | 1–12 |
| `Year` | Calendar year |
| `WeekOfYear` | 1–52 |
| `IsWeekend` | Binary (Sat/Sun = 1) |

---

## 5. Anomaly Detection Assumptions

- **Method:** Per-store Z-score on daily sales.
- **Threshold:** `|z| > 2.5` (configurable via `ANOMALY_ZSCORE` in `config.py`).
- **Column added:** `is_anomaly` (binary), `sales_zscore` (float).
- **Use:** Dashboard anomaly alerts and data quality monitoring — anomalies are **not** removed from training data (they represent real events like Christmas or store fires).

---

## 6. Store Clustering Assumptions

- **Algorithm:** KMeans, `K=4` clusters (configurable via `N_CLUSTERS`).
- **Features used:** `avg_sales`, `avg_customers`, `sales_std`, `promo_frequency`.
- **Normalisation:** StandardScaler applied before KMeans.
- **Labels:** Clusters assigned human-readable labels (High/Mid/Low Performer) based on average sales ranking.

---

## 7. Known Limitations

| Limitation | Impact | Mitigation |
| :--- | :--- | :--- |
| `CompetitionDistance` NaN filled with median | Slightly biases stores with unknown competition toward "average" competition | Acceptable — better than dropping ~10% of stores |
| Lag features create NaN in first 28 rows per store | Those rows are dropped before model training | Use `.dropna()` on lag columns before training |
| External events (COVID, economic shocks) not in data | Model cannot forecast black-swan events | Out of scope for this dataset |
| `train.csv` ends 2015-07-31 — no real future data | Forecasts are extrapolations into a "future" that is now 10 years past | Acceptable for capstone demonstration |

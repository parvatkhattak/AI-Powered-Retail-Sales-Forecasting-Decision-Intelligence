"""
tests/test_data_pipeline.py
Owner: Himanshu — Data Engineer

Checks:
- Raw data loads correctly
- Merge row count is preserved
- Clean data has no nulls in critical columns
- Lag/rolling features contain no data leakage
- Open==0 rows are removed
- Feature columns exist after engineering
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_CSV, STORE_CSV


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_data():
    if not TRAIN_CSV.exists() or not STORE_CSV.exists():
        pytest.skip("Raw CSV files not present — run data pipeline first.")
    from src.data_pipeline import load_raw_data
    return load_raw_data()


@pytest.fixture(scope="module")
def merged_df(raw_data):
    from src.data_pipeline import merge_datasets
    train, store = raw_data
    return merge_datasets(train, store)


@pytest.fixture(scope="module")
def cleaned_df(merged_df):
    from src.data_pipeline import clean_data
    return clean_data(merged_df)


@pytest.fixture(scope="module")
def engineered_df(cleaned_df):
    from src.data_pipeline import engineer_features
    return engineer_features(cleaned_df)


# ── 1. Load ────────────────────────────────────────────────────────────────────

def test_load_returns_two_dataframes(raw_data):
    train, store = raw_data
    assert isinstance(train, pd.DataFrame)
    assert isinstance(store, pd.DataFrame)


def test_train_has_required_columns(raw_data):
    train, _ = raw_data
    for col in ["Store", "Date", "Sales", "Open", "Promo"]:
        assert col in train.columns, f"Missing column: {col}"


def test_store_has_required_columns(raw_data):
    _, store = raw_data
    for col in ["Store", "StoreType", "Assortment", "CompetitionDistance"]:
        assert col in store.columns, f"Missing column: {col}"


def test_train_row_count_nonzero(raw_data):
    train, _ = raw_data
    assert len(train) > 100_000, "train.csv seems too small"


# ── 2. Merge ───────────────────────────────────────────────────────────────────

def test_merge_preserves_row_count(raw_data, merged_df):
    train, _ = raw_data
    assert len(merged_df) == len(train), "Row count changed after merge"


def test_merge_adds_store_columns(merged_df):
    for col in ["StoreType", "Assortment", "CompetitionDistance"]:
        assert col in merged_df.columns, f"Merge missing column: {col}"


def test_merge_no_all_null_rows(merged_df):
    assert not merged_df.isnull().all(axis=1).any(), "Some rows are entirely null after merge"


# ── 3. Clean ───────────────────────────────────────────────────────────────────

def test_clean_removes_closed_days(cleaned_df):
    """All rows in cleaned data must have Open == 1 and Sales > 0."""
    if "Open" in cleaned_df.columns:
        assert (cleaned_df["Open"] == 1).all(), "Closed store rows (Open==0) still present"
    assert (cleaned_df["Sales"] > 0).all(), "Zero-sales rows still present after cleaning"


def test_clean_competition_distance_no_nulls(cleaned_df):
    assert cleaned_df["CompetitionDistance"].isnull().sum() == 0, \
        "CompetitionDistance still has nulls after fillna"


def test_clean_promo2_cols_no_nulls(cleaned_df):
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        if col in cleaned_df.columns:
            assert cleaned_df[col].isnull().sum() == 0, f"{col} has nulls after fill"


def test_clean_reduces_row_count(raw_data, cleaned_df):
    train, _ = raw_data
    assert len(cleaned_df) < len(train), "Cleaning should remove some rows (closed days)"


def test_clean_state_holiday_no_raw_zero(cleaned_df):
    """StateHoliday '0' or '0.0' should be replaced with 'no_holiday'."""
    if "StateHoliday" in cleaned_df.columns:
        assert "0" not in cleaned_df["StateHoliday"].values, "Raw '0' still in StateHoliday"
        assert "0.0" not in cleaned_df["StateHoliday"].values, "Raw '0.0' still in StateHoliday"


# ── 4. Feature Engineering — no data leakage ──────────────────────────────────

def test_lag_columns_exist(engineered_df):
    for lag in [7, 14, 28]:
        col = f"Sales_lag_{lag}"
        assert col in engineered_df.columns, f"Missing lag column: {col}"


def test_rolling_columns_exist(engineered_df):
    for window in [7, 14, 28]:
        for stat in ["mean", "std"]:
            col = f"Sales_roll_{stat}_{window}"
            assert col in engineered_df.columns, f"Missing rolling column: {col}"


def test_lag7_uses_past_data_only(engineered_df):
    """
    For any store, the value of Sales_lag_7 on row i must equal
    the Sales on the row 7 rows prior (within the same store).
    Verifies .shift() was applied per-store, not globally.
    """
    store = engineered_df["Store"].iloc[0]
    store_df = engineered_df[engineered_df["Store"] == store].sort_values("Date").reset_index(drop=True)
    idx = 14  # skip the first 14 rows where lag is NaN
    expected = store_df["Sales"].iloc[idx - 7]
    actual = store_df["Sales_lag_7"].iloc[idx]
    assert abs(actual - expected) < 1e-3, \
        f"Lag-7 leakage detected: expected {expected}, got {actual}"


def test_date_features_exist(engineered_df):
    for col in ["DayOfWeek", "Month", "Year", "WeekOfYear"]:
        assert col in engineered_df.columns, f"Missing date feature: {col}"


def test_no_future_sales_in_features(engineered_df):
    """Sales column itself must not appear as a feature (only as target)."""
    feature_cols = [c for c in engineered_df.columns
                    if c.startswith("Sales_") and c != "Sales"]
    for col in feature_cols:
        # A lag/roll column should never equal the same-row Sales (would be leakage)
        # Just confirm these are not perfect predictors of the current Sales
        corr = engineered_df[col].corr(engineered_df["Sales"])
        assert corr < 1.0, f"{col} is a perfect predictor — likely leakage"

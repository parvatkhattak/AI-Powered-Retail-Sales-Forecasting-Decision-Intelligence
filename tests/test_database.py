"""
tests/test_database.py
Owner: Himanshu — Data Engineer

Checks:
- All query functions return correct column names and data types
- Functions handle edge cases (unknown store, empty results)
- Mock mode returns expected shape
"""
import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, USE_MOCKS


# ── Skip all DB tests if DB is not built and mocks are off ────────────────────

db_available = DB_PATH.exists() or USE_MOCKS


@pytest.fixture(scope="module", autouse=True)
def require_db():
    if not db_available:
        pytest.skip("retail.db not found and USE_MOCKS=False — run data pipeline first.")


# ── get_store_metrics ─────────────────────────────────────────────────────────

class TestGetStoreMetrics:
    def test_returns_dataframe(self):
        from src.database import get_store_metrics
        df = get_store_metrics([1], days=30)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        from src.database import get_store_metrics
        df = get_store_metrics([1], days=30)
        for col in ["store_id", "date", "sales"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_sales_all_positive(self):
        from src.database import get_store_metrics
        df = get_store_metrics([1], days=30)
        if not df.empty:
            assert (df["sales"] > 0).all(), "Some sales are zero or negative"

    def test_empty_store_list_returns_empty(self):
        from src.database import get_store_metrics
        df = get_store_metrics([], days=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_multiple_stores(self):
        from src.database import get_store_metrics
        df = get_store_metrics([1, 2, 3], days=30)
        assert isinstance(df, pd.DataFrame)


# ── get_promo_history ─────────────────────────────────────────────────────────

class TestGetPromoHistory:
    def test_returns_dict(self):
        from src.database import get_promo_history
        result = get_promo_history(1)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from src.database import get_promo_history
        result = get_promo_history(1)
        for key in ["promo_avg_sales", "non_promo_avg_sales", "uplift_pct"]:
            assert key in result, f"Missing key: {key}"

    def test_uplift_is_numeric(self):
        from src.database import get_promo_history
        result = get_promo_history(1)
        assert isinstance(result["uplift_pct"], (int, float))


# ── get_eda_summary ───────────────────────────────────────────────────────────

class TestGetEdaSummary:
    def test_returns_dict(self):
        from src.database import get_eda_summary
        result = get_eda_summary()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from src.database import get_eda_summary
        result = get_eda_summary()
        for key in ["total_stores", "avg_daily_sales", "best_store_id", "worst_store_id"]:
            assert key in result, f"Missing key: {key}"

    def test_total_stores_positive(self):
        from src.database import get_eda_summary
        result = get_eda_summary()
        assert result["total_stores"] > 0

    def test_avg_sales_positive(self):
        from src.database import get_eda_summary
        result = get_eda_summary()
        assert result["avg_daily_sales"] > 0


# ── get_sales_trend ───────────────────────────────────────────────────────────

class TestGetSalesTrend:
    def test_returns_dataframe(self):
        from src.database import get_sales_trend
        df = get_sales_trend(1, period="monthly")
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        from src.database import get_sales_trend
        df = get_sales_trend(1, period="monthly")
        for col in ["period", "total_sales", "avg_sales"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_supports_all_periods(self):
        from src.database import get_sales_trend
        for period in ["daily", "weekly", "monthly"]:
            df = get_sales_trend(1, period=period)
            assert isinstance(df, pd.DataFrame), f"Failed for period={period}"


# ── get_all_stores ────────────────────────────────────────────────────────────

class TestGetAllStores:
    def test_returns_dataframe(self):
        from src.database import get_all_stores
        df = get_all_stores()
        assert isinstance(df, pd.DataFrame)

    def test_has_store_column(self):
        from src.database import get_all_stores
        df = get_all_stores()
        assert "Store" in df.columns

    def test_no_duplicate_stores(self):
        from src.database import get_all_stores
        df = get_all_stores()
        assert df["Store"].nunique() == len(df), "Duplicate stores returned"


# ── get_promo_uplift_ranking ──────────────────────────────────────────────────

class TestGetPromoUpliftRanking:
    def test_returns_dataframe(self):
        from src.database import get_promo_uplift_ranking
        df = get_promo_uplift_ranking(top_n=10)
        assert isinstance(df, pd.DataFrame)

    def test_respects_top_n(self):
        from src.database import get_promo_uplift_ranking
        df = get_promo_uplift_ranking(top_n=5)
        assert len(df) <= 5

    def test_has_uplift_pct_column(self):
        from src.database import get_promo_uplift_ranking
        df = get_promo_uplift_ranking(top_n=10)
        assert "uplift_pct" in df.columns

    def test_uplift_sorted_descending(self):
        from src.database import get_promo_uplift_ranking
        df = get_promo_uplift_ranking(top_n=10)
        if len(df) > 1:
            assert df["uplift_pct"].is_monotonic_decreasing or \
                   (df["uplift_pct"].diff().dropna() <= 0).all()


# ── get_competition_impact_analysis ──────────────────────────────────────────

class TestGetCompetitionImpactAnalysis:
    def test_returns_dataframe(self):
        from src.database import get_competition_impact_analysis
        df = get_competition_impact_analysis()
        assert isinstance(df, pd.DataFrame)

    def test_has_distance_bucket_column(self):
        from src.database import get_competition_impact_analysis
        df = get_competition_impact_analysis()
        assert "distance_bucket" in df.columns


# ── get_anomaly_flags ─────────────────────────────────────────────────────────

class TestGetAnomalyFlags:
    def test_returns_dataframe(self):
        from src.database import get_anomaly_flags
        df = get_anomaly_flags(1, lookback_days=90)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        from src.database import get_anomaly_flags
        df = get_anomaly_flags(1, lookback_days=90)
        for col in ["date", "sales", "sales_zscore"]:
            assert col in df.columns, f"Missing column: {col}"

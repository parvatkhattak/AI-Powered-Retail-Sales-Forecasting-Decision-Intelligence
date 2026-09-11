"""
tests/test_model_engine.py
Owner: Ashutosh — ML Engineer

Checks:
- 7-day forecast returns exactly 7 rows with positive values
- SHAP returns top 5 features with correct structure
- SHAP waterfall data is valid
- What-If forecast changes when promo is toggled
- Model metrics contain required keys
- Baseline comparison returns a dict
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LGBM_PATH, USE_MOCKS, DB_PATH


# ── Skip if models or DB not present and not in mock mode ─────────────────────

@pytest.fixture(scope="module", autouse=True)
def require_artifacts():
    if not USE_MOCKS and not LGBM_PATH.exists():
        pytest.skip("Trained model not found — run compare_models.py or train_model() first.")


# ── get_7day_forecast ─────────────────────────────────────────────────────────

class TestGet7DayForecast:
    def test_returns_dataframe(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1)
        assert isinstance(df, pd.DataFrame)

    def test_exactly_7_rows(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1)
        assert len(df) == 7, f"Expected 7 forecast rows, got {len(df)}"

    def test_all_sales_positive(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1)
        assert (df["predicted_sales"] > 0).all(), "Forecast contains zero or negative values"

    def test_has_required_columns(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1)
        for col in ["date", "predicted_sales"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_dates_are_consecutive(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1).sort_values("date")
        dates = pd.to_datetime(df["date"])
        diffs = dates.diff().dropna()
        assert (diffs == pd.Timedelta(days=1)).all(), "Forecast dates are not consecutive daily"

    def test_works_for_any_valid_store(self):
        from src.model_engine import get_7day_forecast
        for store_id in [1, 100, 500, 1115]:
            df = get_7day_forecast(store_id)
            assert len(df) == 7, f"Store {store_id}: expected 7 rows, got {len(df)}"

    def test_has_confidence_bounds(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1)
        assert "lower_bound" in df.columns or "upper_bound" in df.columns, \
            "Forecast should include confidence interval bounds"


# ── get_shap_explanations ─────────────────────────────────────────────────────

class TestGetShapExplanations:
    def test_returns_list(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        assert isinstance(result, list), "SHAP explanations should be a list"

    def test_returns_at_least_5_features(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        assert len(result) >= 5, f"Expected at least 5 SHAP features, got {len(result)}"

    def test_each_item_has_required_keys(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        for item in result[:5]:
            assert "feature" in item, "SHAP item missing 'feature' key"
            assert "importance" in item or "shap_value" in item, \
                "SHAP item missing importance/shap_value key"

    def test_features_are_strings(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        for item in result:
            assert isinstance(item.get("feature", ""), str)


# ── get_shap_waterfall_data ───────────────────────────────────────────────────

class TestGetShapWaterfallData:
    def test_returns_dict_or_list(self):
        from src.model_engine import get_shap_waterfall_data
        result = get_shap_waterfall_data(1)
        assert isinstance(result, (dict, list)), "Waterfall data should be dict or list"

    def test_nonempty(self):
        from src.model_engine import get_shap_waterfall_data
        result = get_shap_waterfall_data(1)
        assert result, "Waterfall data is empty"


# ── get_whatif_forecast ───────────────────────────────────────────────────────

class TestGetWhatIfForecast:
    def test_returns_dataframe(self):
        from src.model_engine import get_whatif_forecast
        df = get_whatif_forecast(1, promo=1)
        assert isinstance(df, pd.DataFrame)

    def test_exactly_7_rows(self):
        from src.model_engine import get_whatif_forecast
        df = get_whatif_forecast(1, promo=1)
        assert len(df) == 7

    def test_promo_changes_forecast(self):
        """Toggling promo should produce different total sales."""
        from src.model_engine import get_whatif_forecast
        df_promo = get_whatif_forecast(1, promo=1)
        df_no_promo = get_whatif_forecast(1, promo=0)
        total_promo = df_promo["predicted_sales"].sum()
        total_no_promo = df_no_promo["predicted_sales"].sum()
        # They don't have to be vastly different, but they must not be identical
        assert total_promo != total_no_promo, \
            "What-If: promo toggle had no effect on forecast — check implementation"

    def test_promo_forecast_higher_than_no_promo(self):
        """Promo days should generally forecast higher sales."""
        from src.model_engine import get_whatif_forecast
        df_promo = get_whatif_forecast(1, promo=1)
        df_no_promo = get_whatif_forecast(1, promo=0)
        assert df_promo["predicted_sales"].mean() >= df_no_promo["predicted_sales"].mean() * 0.95, \
            "Promo forecast should be >= no-promo forecast (allowing 5% tolerance)"


# ── get_model_metrics ─────────────────────────────────────────────────────────

class TestGetModelMetrics:
    def test_returns_dict(self):
        from src.model_engine import get_model_metrics
        result = get_model_metrics()
        assert isinstance(result, dict)

    def test_has_lightgbm_key(self):
        from src.model_engine import get_model_metrics
        result = get_model_metrics()
        assert "lightgbm" in result, "Metrics missing lightgbm section"

    def test_rmspe_below_threshold(self):
        """LightGBM RMSPE should be under 0.20 (our target is ~0.12)."""
        from src.model_engine import get_model_metrics
        result = get_model_metrics()
        lgbm_rmspe = result["lightgbm"]["rmspe"]
        assert lgbm_rmspe < 0.20, f"LightGBM RMSPE too high: {lgbm_rmspe:.4f}"

    def test_primary_model_is_lightgbm(self):
        from src.model_engine import get_model_metrics
        result = get_model_metrics()
        assert result.get("primary_model") == "lightgbm"


# ── get_baseline_comparison ───────────────────────────────────────────────────

class TestGetBaselineComparison:
    def test_returns_dict(self):
        from src.model_engine import get_baseline_comparison
        result = get_baseline_comparison(1)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from src.model_engine import get_baseline_comparison
        result = get_baseline_comparison(1)
        for key in ["baseline_rmspe", "model_rmspe", "improvement_pct"]:
            assert key in result, f"Missing key: {key}"

    def test_model_beats_baseline(self):
        from src.model_engine import get_baseline_comparison
        result = get_baseline_comparison(1)
        assert result["model_rmspe"] < result["baseline_rmspe"], \
            "Model should outperform baseline"

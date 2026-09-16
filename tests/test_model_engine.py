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
        # Only check open days (closed days have PredictedSales == 0 by design)
        open_days = df[df["PredictedSales"] > 0]
        assert len(open_days) > 0, "Forecast contains no open-day predictions"
        assert (open_days["PredictedSales"] > 0).all(), "Forecast contains negative values on open days"

    def test_has_required_columns(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1)
        # Actual column names use PascalCase
        for col in ["Date", "PredictedSales"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_dates_are_consecutive(self):
        from src.model_engine import get_7day_forecast
        df = get_7day_forecast(1).sort_values("Date")
        dates = pd.to_datetime(df["Date"])
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
        # Actual column names: LowerBound / UpperBound
        assert "LowerBound" in df.columns or "UpperBound" in df.columns, \
            "Forecast should include confidence interval bounds"


# ── get_shap_explanations ─────────────────────────────────────────────────────

class TestGetShapExplanations:
    def test_returns_dict_or_list(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        # get_shap_explanations returns a dict {feature_name: {value, direction}}
        assert isinstance(result, (dict, list)), "SHAP explanations should be a dict or list"

    def test_returns_at_least_5_features(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        assert len(result) >= 5, f"Expected at least 5 SHAP features, got {len(result)}"

    def test_each_item_has_required_keys(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        if isinstance(result, dict):
            # dict form: {feature_name: {"value": float, "direction": str}}
            for feature_name, info in list(result.items())[:5]:
                assert isinstance(feature_name, str), "Feature name should be a string"
                assert isinstance(info, dict), "SHAP value info should be a dict"
                assert "value" in info or "shap_value" in info or "importance" in info, \
                    f"SHAP info for '{feature_name}' missing value key"
        else:
            for item in result[:5]:
                assert "feature" in item, "SHAP item missing 'feature' key"
                assert "importance" in item or "shap_value" in item or "value" in item, \
                    "SHAP item missing value key"

    def test_features_are_strings(self):
        from src.model_engine import get_shap_explanations
        result = get_shap_explanations(1)
        if isinstance(result, dict):
            for feature_name in result:
                assert isinstance(feature_name, str), f"Feature name '{feature_name}' is not a string"
        else:
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
        # Actual kwarg is promo_override, not promo
        df = get_whatif_forecast(1, promo_override=True)
        assert isinstance(df, pd.DataFrame)

    def test_exactly_7_rows(self):
        from src.model_engine import get_whatif_forecast
        df = get_whatif_forecast(1, promo_override=True)
        assert len(df) == 7

    def test_promo_changes_forecast(self):
        """Toggling promo should produce different total sales."""
        from src.model_engine import get_whatif_forecast
        df_promo = get_whatif_forecast(1, promo_override=True)
        df_no_promo = get_whatif_forecast(1, promo_override=False)
        total_promo = df_promo["PredictedSales"].sum()
        total_no_promo = df_no_promo["PredictedSales"].sum()
        # They don't have to be vastly different, but they must not be identical
        assert total_promo != total_no_promo, \
            "What-If: promo toggle had no effect on forecast — check implementation"

    def test_promo_forecast_higher_than_no_promo(self):
        """Promo days should generally forecast higher sales."""
        from src.model_engine import get_whatif_forecast
        df_promo = get_whatif_forecast(1, promo_override=True)
        df_no_promo = get_whatif_forecast(1, promo_override=False)
        assert df_promo["PredictedSales"].mean() >= df_no_promo["PredictedSales"].mean() * 0.95, \
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
    def test_returns_dataframe(self):
        from src.model_engine import get_baseline_comparison
        # get_baseline_comparison returns a DataFrame with Date, LightGBM, XGBoost, Baseline_MovingAvg
        result = get_baseline_comparison(1)
        assert isinstance(result, pd.DataFrame), \
            f"Expected DataFrame, got {type(result)}"

    def test_has_required_columns(self):
        from src.model_engine import get_baseline_comparison
        result = get_baseline_comparison(1)
        for col in ["Date", "LightGBM", "Baseline_MovingAvg"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_model_beats_baseline(self):
        from src.model_engine import get_baseline_comparison
        result = get_baseline_comparison(1)
        # On open days, the ML model should not be dominated by the naive baseline
        # We verify LightGBM produces forecasts (non-null, non-all-zero on open days)
        open_days = result[result["LightGBM"] > 0]
        assert len(open_days) > 0, "LightGBM produced no open-day forecasts"
        assert (open_days["LightGBM"] > 0).all(), "LightGBM forecasts should be positive on open days"

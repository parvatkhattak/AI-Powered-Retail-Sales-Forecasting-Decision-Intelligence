"""
src/model_engine.py
Owner: Dev 3 — ML Engineer

Responsibilities:
- Train XGBoost and LightGBM models
- Perform walk-forward cross-validation
- Provide 7-day forecasts and SHAP explanations to the app
- Handle mock outputs if USE_MOCKS is True
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import USE_MOCKS, MOCK_FORECAST, MOCK_SHAP

def train_model() -> None:
    """Trains the forecasting models and saves artifacts to models/."""
    # TODO (Dev 3): Implement training pipeline
    print("MOCK: Model training stub")

def get_7day_forecast(store_id: int) -> pd.DataFrame:
    """Returns a 7-row DataFrame: Date, PredictedSales, LowerBound, UpperBound."""
    if USE_MOCKS:
        return pd.DataFrame([{"date": "2015-08-01", "predicted_sales": 6500, "lower": 6000, "upper": 7000}])
    # TODO (Dev 3): Implement real forecast
    return pd.DataFrame()

def get_shap_explanations(store_id: int) -> dict:
    """Returns top 5 SHAP drivers: {feature_name: {shap_value, direction}}."""
    if USE_MOCKS:
        return {"Promo": {"value": 1500, "direction": "positive"}}
    # TODO (Dev 3): Implement real SHAP
    return {}

def get_model_metrics() -> dict:
    """Returns overall model evaluation metrics."""
    # TODO (Dev 3): Implement
    return {}

def get_baseline_comparison(store_id: int) -> pd.DataFrame:
    """Returns forecast from baseline vs ML model."""
    # TODO (Dev 3): Implement
    return pd.DataFrame()

def get_whatif_forecast(store_id: int, promo_override: bool) -> pd.DataFrame:
    """Returns forecast with promo forced on/off."""
    # TODO (Dev 3): Implement (S-Grade)
    return pd.DataFrame()

def get_shap_waterfall_data(store_id: int) -> dict:
    """Returns full SHAP waterfall data."""
    # TODO (Dev 3): Implement (S-Grade)
    return {}

if __name__ == "__main__":
    if "--train" in sys.argv:
        train_model()

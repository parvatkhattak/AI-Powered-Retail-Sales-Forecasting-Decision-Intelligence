"""
src/feature_engineering.py
Owner: Dev 3 — ML Engineer (with Dev 2)

Responsibilities:
- Create time-series and categorical features for modelling
- Ensure no data leakage (proper shifting of lags)
"""

import pandas as pd
import numpy as np

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO (Dev 3): Implement
    return df

def add_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    # TODO (Dev 3): Implement
    return df

def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    # TODO (Dev 3): Implement
    return df

def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO (Dev 3): Implement
    return df

def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO (Dev 3): Implement
    return df

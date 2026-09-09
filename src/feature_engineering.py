"""
src/feature_engineering.py
Owner: Ashutosh — ML Engineer (with Himanshu)

Responsibilities:
- Create time-series and categorical features for modelling
- Ensure no data leakage (proper shifting of lags)
"""

import pandas as pd
import numpy as np

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    for lag in lag_days:
        df[f"Sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    shifted = df.groupby("Store")["Sales"].shift(1)  # yesterday and earlier only
    for w in windows:
        df[f"Sales_roll_mean_{w}"] = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"Sales_roll_std_{w}"]  = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(w, min_periods=1).std())
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:

    # with 0 (not NaN) — so 0 here means "unknown", not a real month/year. Treat it as missing.
    since_year = df["CompetitionOpenSinceYear"].replace(0, np.nan)
    since_month = df["CompetitionOpenSinceMonth"].replace(0, np.nan)

    comp_open_date = pd.to_datetime(
        since_year.astype("Int64").astype(str) + "-" +
        since_month.astype("Int64").astype(str) + "-01",
        errors="coerce"
    )

    months_since = (
        (df["Date"].dt.year - comp_open_date.dt.year) * 12 +
        (df["Date"].dt.month - comp_open_date.dt.month)
    )
    # negative = competitor's official "open date" is after this row's date,
    # i.e. it hadn't opened yet at this point in time -> treat as "0 months of competition"
    df["CompetitionOpenMonths"] = months_since.clip(lower=0)

    # raw month/year are now redundant — the one clean column above replaces both
    df = df.drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"])

    return df


def _promo_streak(promo_series: pd.Series) -> pd.Series:
    """Consecutive days Promo has been running, ending at each row (0 if not in a promo)."""
    blocks = (promo_series != promo_series.shift()).cumsum()
    streak = promo_series.groupby(blocks).cumcount() + 1
    return streak * promo_series


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # --- IsPromo2Active: Promo2 + which months it runs + has it actually started yet ---
    since_year = df["Promo2SinceYear"].replace(0, np.nan)
    since_week = df["Promo2SinceWeek"].replace(0, np.nan)
    valid = since_year.notna() & since_week.notna()
    promo2_start = pd.Series(pd.NaT, index=df.index)
    promo2_start[valid] = [
        pd.Timestamp.fromisocalendar(int(y), int(w), 1)
        for y, w in zip(since_year[valid], since_week[valid])
    ]

    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sept",10:"Oct",11:"Nov",12:"Dec"}
    month_abbr = df["Date"].dt.month.map(month_map)
    in_interval = [
        m in interval.split(",") if interval else False
        for m, interval in zip(month_abbr, df["PromoInterval"])
    ]
    df["IsPromo2Active"] = (
        (df["Promo2"] == 1) & (df["Date"] >= promo2_start) & pd.Series(in_interval, index=df.index)
    ).astype(int)

    # --- Promo streak: how many consecutive days this promo has been running ---
    df["PromoStreak"] = df.groupby("Store")["Promo"].transform(_promo_streak)

    # --- Days since the last promo ended, per store ---
    last_promo_date = df["Date"].where(df["Promo"] == 1).groupby(df["Store"]).ffill()
    df["DaysSinceLastPromo"] = (df["Date"] - last_promo_date).dt.days

    # --- Days until the next promo starts (legitimate — Promo is planned in advance,
    #     same reason it's already given to you in test.csv, not something you're predicting) ---
    next_promo_date = df["Date"].where(df["Promo"] == 1).groupby(df["Store"]).bfill()
    df["DaysUntilNextPromo"] = (next_promo_date - df["Date"]).dt.days

    # raw/now-redundant columns — including your friend's IsPromoMonth (see note below)
    df = df.drop(columns=["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "IsPromoMonth"])

    return df

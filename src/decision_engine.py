"""
src/decision_engine.py
Owner: Saumya — Agentic AI Engineer

Responsibilities:
- Structure the observation, prediction, evidence, and recommendation
- Rank stores for comparison

Deterministic by design: same underlying data always produces the same
risk score and ranking, so the LLM never has to invent a number — it only
formats what this module already computed (see docs/architecture.md, 6.4).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import database, model_engine

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


def _trend_pct(metrics_df) -> float:
    """% change between the first and second half of the sales window."""
    if metrics_df.empty or len(metrics_df) < 4:
        return 0.0
    sales = metrics_df.sort_values("date")["sales"]
    midpoint = len(sales) // 2
    first_half, second_half = sales.iloc[:midpoint], sales.iloc[midpoint:]
    if first_half.mean() == 0:
        return 0.0
    return round((second_half.mean() - first_half.mean()) / first_half.mean() * 100, 2)


def _days_since_last_promo(metrics_df) -> int | None:
    if metrics_df.empty or "promo" not in metrics_df.columns:
        return None
    promo_rows = metrics_df[metrics_df["promo"] == 1]
    if promo_rows.empty:
        return len(metrics_df)  # no promo at all in the window
    last_promo_date = promo_rows["date"].max()
    return int((metrics_df["date"].max() - last_promo_date).days)


def _risk_score(trend_pct: float, forecast_vs_avg_pct: float, days_since_promo: int | None) -> tuple[str, float]:
    """Composite score across trend, forecast weakness, and unused promo potential."""
    score = 0.0
    if trend_pct < 0:
        score += min(abs(trend_pct), 40)
    if forecast_vs_avg_pct < 0:
        score += min(abs(forecast_vs_avg_pct), 40)
    if days_since_promo and days_since_promo > 14:
        score += 20
    level = "HIGH" if score >= 50 else "MEDIUM" if score >= 20 else "LOW"
    return level, round(score, 2)


def generate_decision_report(store_id: int) -> dict:
    """Returns full Observation/Prediction/Evidence/Recommendation for a store."""
    metrics = database.get_store_metrics([store_id], days=30)
    promo = database.get_promo_history(store_id)
    forecast = model_engine.get_7day_forecast(store_id)
    shap = model_engine.get_shap_explanations(store_id)

    sources = ["database.get_store_metrics", "database.get_promo_history"]

    trend_pct = _trend_pct(metrics)
    avg_sales = round(float(metrics["sales"].mean()), 2) if not metrics.empty else 0.0

    if not forecast.empty:
        sources.append("model_engine.get_7day_forecast")
        forecast_avg = round(float(forecast["PredictedSales"].mean()), 2)
        forecast_vs_avg_pct = round((forecast_avg - avg_sales) / avg_sales * 100, 2) if avg_sales else 0.0
    else:
        forecast_avg, forecast_vs_avg_pct = 0.0, 0.0

    if shap:
        sources.append("model_engine.get_shap_explanations")
        top_driver_name, top_driver = max(shap.items(), key=lambda kv: abs(kv[1]["value"]))
    else:
        top_driver_name, top_driver = None, None

    days_since_promo = _days_since_last_promo(metrics)
    risk_level, risk_score = _risk_score(trend_pct, forecast_vs_avg_pct, days_since_promo)

    if metrics.empty:
        observation = f"No historical sales data available for Store {store_id}."
    else:
        direction = "declined" if trend_pct < 0 else "grown"
        observation = (
            f"Store {store_id} sales have {direction} {abs(trend_pct)}% over the last "
            f"{len(metrics)} days (avg €{avg_sales:,.0f}/day)."
        )

    if forecast.empty:
        prediction = f"No forecast available for Store {store_id}."
    else:
        sign = "+" if forecast_vs_avg_pct >= 0 else ""
        prediction = (
            f"The model forecasts Store {store_id} will average €{forecast_avg:,.0f}/day next week "
            f"({sign}{forecast_vs_avg_pct}% vs its own 30-day average)."
        )

    uplift_pct = promo.get("uplift_pct", 0)
    if top_driver_name:
        evidence = (
            f"SHAP analysis shows '{top_driver_name}' is the strongest driver of next-day sales "
            f"({top_driver['direction']}, impact {top_driver['value']:+.2f}). "
            f"Promo uplift for this store is {uplift_pct}% "
            f"(promo avg €{promo.get('promo_avg_sales', 0):,.0f} vs non-promo €{promo.get('non_promo_avg_sales', 0):,.0f})."
        )
    else:
        evidence = f"Promo uplift for this store is {uplift_pct}%."

    if risk_level == "HIGH" and days_since_promo and days_since_promo > 14 and uplift_pct > 0:
        recommendation = (
            f"Priority: Store {store_id}. Activate a promotion next week — a historical uplift of "
            f"{uplift_pct}% and {days_since_promo} days since the last promo support this action."
        )
    elif risk_level == "HIGH":
        recommendation = f"Priority: Store {store_id}. Investigate the sales decline — both trend and forecast are weak."
    else:
        recommendation = f"Store {store_id} is stable — no urgent action needed this week."

    return {
        "store_id": store_id,
        "observation": observation,
        "prediction": prediction,
        "evidence": evidence,
        "recommendation": recommendation,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "trend_pct": trend_pct,
        "forecast_avg_sales": forecast_avg,
        "data_sources": sources,
    }


def compare_stores_report(store_ids: list[int]) -> dict:
    """Returns ranked priority list for multiple stores."""
    reports = [generate_decision_report(sid) for sid in store_ids]
    ranked = sorted(reports, key=lambda r: r["risk_score"], reverse=True)

    if not ranked:
        return {"ranked_stores": [], "summary": "No stores to compare."}

    top = ranked[0]
    summary = f"Priority: Store {top['store_id']}. {top['recommendation']}"
    return {"ranked_stores": ranked, "summary": summary}

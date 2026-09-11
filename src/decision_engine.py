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


# Each risk level maps to exactly one urgency, and each urgency to one shape of
# advice. Keeping the mapping in one table is what stops a store being called
# "MEDIUM risk" and "stable — no urgent action needed" in the same breath: the
# recommendation is derived from the level rather than written independently of it.
URGENCY_BY_LEVEL = {"HIGH": "act_now", "MEDIUM": "monitor", "LOW": "none"}
ACTION_REQUIRED = {"act_now": True, "monitor": True, "none": False}


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


def _risk_drivers(trend_pct: float, forecast_vs_avg_pct: float, days_since_promo: int | None) -> list[str]:
    """Which signals actually pushed the score up — so the recommendation can
    name the reason instead of asserting a risk level with nothing behind it."""
    drivers = []
    if trend_pct < 0:
        drivers.append(f"sales trending down {abs(trend_pct)}% over the window")
    if forecast_vs_avg_pct < 0:
        drivers.append(f"next week forecast {abs(forecast_vs_avg_pct)}% below its own average")
    if days_since_promo and days_since_promo > 14:
        drivers.append(f"{days_since_promo} days since the last promotion")
    return drivers


def _recommendation(store_id: int, risk_level: str, drivers: list[str],
                    days_since_promo: int | None, uplift_pct: float) -> str:
    """The advice for one store, derived from its risk level.

    MEDIUM used to fall into the same branch as LOW and produce "stable — no
    urgent action needed", which then got quoted as the priority store's
    recommendation in a comparison. A MEDIUM store is not stable; it gets a
    watch action, and only LOW gets "nothing to do".
    """
    reason = drivers[0] if drivers else "no negative signals in the current window"

    if risk_level == "HIGH":
        if days_since_promo and days_since_promo > 14 and uplift_pct > 0:
            return (
                f"Act this week on Store {store_id}: schedule a promotion. "
                f"A historical uplift of {uplift_pct}% and {days_since_promo} days since "
                f"the last promo make it the highest-value lever available."
            )
        return (
            f"Act this week on Store {store_id}: investigate the decline — "
            f"{'; '.join(drivers) if drivers else 'trend and forecast are both weak'}."
        )

    if risk_level == "MEDIUM":
        if days_since_promo and days_since_promo > 14 and uplift_pct > 0:
            return (
                f"Watch Store {store_id} this week — {reason}. A promotion is the "
                f"obvious lever if it slips further ({uplift_pct}% historical uplift), "
                f"but nothing here needs emergency action yet."
            )
        return (
            f"Watch Store {store_id} this week — {reason}. Worth a check-in, "
            f"but nothing here needs emergency action yet."
        )

    return f"No action needed for Store {store_id} — no weakening signals in the current window."


def generate_decision_report(store_id: int, with_explanations: bool = True) -> dict:
    """Returns full Observation/Prediction/Evidence/Recommendation for a store.

    `with_explanations=False` skips the SHAP call, which is the single most
    expensive part of a report (~0.5s per store) and feeds only the Evidence
    prose — `_risk_score` never looks at it. Fleet-wide screening uses that to
    score a shortlist cheaply, then re-runs the handful it actually shows with
    explanations on. The score is identical either way.
    """
    metrics = database.get_store_metrics([store_id], days=30)
    promo = database.get_promo_history(store_id)
    forecast = model_engine.get_7day_forecast(store_id)
    shap = model_engine.get_shap_explanations(store_id) if with_explanations else {}

    sources = ["database.get_store_metrics", "database.get_promo_history"]

    trend_pct = _trend_pct(metrics)
    avg_sales = round(float(metrics["sales"].mean()), 2) if not metrics.empty else 0.0

    # A closed day forecasts as 0, correctly — the store sells nothing when it
    # is shut. But data_pipeline.py drops Open == 0 rows, so `avg_sales` above
    # is a *trading-day* average with no zeros in it. Averaging the forecast
    # across all 7 calendar days compared a week containing a closed Sunday
    # against a history containing none, which pushed roughly one day in seven
    # (~14%) of phantom weakness into every store's forecast delta and from
    # there straight into its risk score. Both sides are trading days now.
    closed_days = 0
    if not forecast.empty:
        sources.append("model_engine.get_7day_forecast")
        open_days = forecast[forecast["PredictedSales"] > 0]
        closed_days = int(len(forecast) - len(open_days))
        forecast_avg = round(float(open_days["PredictedSales"].mean()), 2) if not open_days.empty else 0.0
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
    drivers = _risk_drivers(trend_pct, forecast_vs_avg_pct, days_since_promo)

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
        closed_note = (
            f" ({closed_days} closed day{'s' if closed_days != 1 else ''} in the window "
            f"excluded — a shut store forecasts zero, which is not underperformance)"
            if closed_days else ""
        )
        prediction = (
            f"The model forecasts Store {store_id} will average €{forecast_avg:,.0f}/day across its "
            f"trading days next week ({sign}{forecast_vs_avg_pct}% vs its own 30-day trading "
            f"average){closed_note}."
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

    urgency = URGENCY_BY_LEVEL[risk_level]
    recommendation = _recommendation(store_id, risk_level, drivers, days_since_promo, uplift_pct)

    return {
        "store_id": store_id,
        "observation": observation,
        "prediction": prediction,
        "evidence": evidence,
        "recommendation": recommendation,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "urgency": urgency,
        "action_required": ACTION_REQUIRED[urgency],
        "risk_drivers": drivers,
        "trend_pct": trend_pct,
        "forecast_avg_sales": forecast_avg,
        "forecast_vs_avg_pct": forecast_vs_avg_pct,
        "forecast_closed_days": closed_days,
        "forecast_available": not forecast.empty,
        "data_sources": sources,
    }


def _comparison_summary(ranked: list[dict]) -> str:
    """One sentence about the group, phrased to match the top store's urgency.

    The old version was f"Priority: Store {id}. {recommendation}", which for a
    non-HIGH store produced "Priority: Store 200. Store 200 is stable — no
    urgent action needed this week." — a ranking and a recommendation flatly
    contradicting each other in one line. The ranking is still the ranking; what
    changes is that a top-of-list store is only called a *priority* when its own
    risk level says action is due.
    """
    top = ranked[0]
    store_id, level, score = top["store_id"], top["risk_level"], top["risk_score"]

    if top["urgency"] == "act_now":
        return f"Priority: Store {store_id} ({level} risk, score {score}). {top['recommendation']}"

    if top["urgency"] == "monitor":
        return (
            f"Nothing in this group needs emergency action this week. Store {store_id} "
            f"ranks first on risk ({level}, score {score}) and is the one to watch — "
            f"{top['risk_drivers'][0] if top['risk_drivers'] else 'it leads the group on risk score'}."
        )

    return (
        f"All {len(ranked)} stores are LOW risk this week — no action needed for any of them. "
        f"Store {store_id} is closest to needing attention (score {score}), so start there "
        f"if you only have time for one."
    )


def compare_stores_report(store_ids: list[int]) -> dict:
    """Returns ranked priority list for multiple stores."""
    reports = [generate_decision_report(sid) for sid in store_ids]
    ranked = sorted(reports, key=lambda r: r["risk_score"], reverse=True)

    if not ranked:
        return {"ranked_stores": [], "summary": "No stores to compare."}

    return {
        "ranked_stores": ranked,
        "summary": _comparison_summary(ranked),
        "action_required": any(r["action_required"] for r in ranked),
        "highest_risk_level": ranked[0]["risk_level"],
    }


# ── Consistency invariant ────────────────────────────────────────────────────

# Phrases that assert nothing needs doing. If one of these appears in the advice
# for a store the engine also ranked as needing action, the report contradicts
# itself and the answer built from it will too.
_NO_ACTION_PHRASES = ("no action needed", "no urgent action", "is stable", "nothing to do")
_ACTION_PHRASES = ("priority:", "act this week", "investigate", "schedule a promotion")


def check_report_consistency(report: dict) -> list[str]:
    """Every way a report could contradict itself, as a list of violations.

    Exposed rather than kept in the tests because the agent runs it on its own
    output too: a report that fails here must never be turned into an answer.
    """
    violations: list[str] = []
    stores = report.get("ranked_stores") or ([report] if "risk_level" in report else [])

    for entry in stores:
        store_id = entry.get("store_id")
        level = entry.get("risk_level")
        urgency = entry.get("urgency")
        text = (entry.get("recommendation") or "").lower()

        if URGENCY_BY_LEVEL.get(level) != urgency:
            violations.append(f"Store {store_id}: risk {level} does not map to urgency {urgency!r}")

        says_no_action = any(p in text for p in _NO_ACTION_PHRASES)
        says_action = any(p in text for p in _ACTION_PHRASES)

        if entry.get("action_required") and says_no_action:
            violations.append(
                f"Store {store_id}: ranked as needing action ({level}) but the "
                f"recommendation says nothing needs doing"
            )
        if not entry.get("action_required") and says_action:
            violations.append(
                f"Store {store_id}: ranked LOW risk but the recommendation "
                f"prescribes an urgent action"
            )
        if says_no_action and says_action:
            violations.append(f"Store {store_id}: recommendation both prescribes and rules out action")

    scores = [s.get("risk_score", 0) for s in stores]
    if report.get("ranked_stores") and scores != sorted(scores, reverse=True):
        violations.append(f"ranking is not ordered by risk score: {scores}")

    summary = (report.get("summary") or "").lower()
    if summary and stores:
        top = stores[0]
        if summary.startswith("priority:") and not top.get("action_required"):
            violations.append(
                f"summary calls Store {top.get('store_id')} a priority while its own "
                f"risk level ({top.get('risk_level')}) says no action is needed"
            )
        if any(p in summary for p in _NO_ACTION_PHRASES) and summary.startswith("priority:"):
            violations.append("summary both names a priority and says no action is needed")

    return violations


# ── Fleet-wide risk ranking ──────────────────────────────────────────────────

# Screening the whole fleet with the full methodology means one forecast per
# store — 1,115 of them, minutes of work. A cheap SQL trend pass narrows the
# field first, and the real methodology scores only the shortlist.
DEFAULT_SCREEN_SIZE = 12


def rank_fleet_risk(top_n: int = 5, screen_size: int = DEFAULT_SCREEN_SIZE) -> dict:
    """The stores most at risk across the whole fleet, using this project's own
    risk methodology rather than a new one.

    Two stages, both honest about what they are:

    1. **Screen** — `database.get_fleet_trend_screen()` ranks all 1,115 stores
       by recent sales trend in one SQL pass and returns the weakest
       `screen_size`. This is a shortlist, not a risk score: a store is a
       *candidate* because its sales are falling, not because it is "worse
       than another store".
    2. **Score** — every candidate goes through `generate_decision_report()`,
       the same trend + forecast-weakness + unused-promo score used everywhere
       else, and the ranking is by that score.

    A store whose forecast is unavailable is kept and flagged rather than
    dropped or given an invented number: its score is computed from the signals
    that *are* available, and the report says so.
    """
    screen = database.get_fleet_trend_screen(limit=max(top_n, screen_size))
    if screen is None or screen.empty:
        return {"ranked_stores": [], "summary": "No store has enough sales history to rank on risk.",
                "screened_stores": 0, "scored_stores": 0, "stores_without_forecast": [],
                "methodology": "no data"}

    candidates = [int(s) for s in screen["Store"].tolist()]
    # Score the shortlist without SHAP, then re-run only the stores that will
    # actually be shown with their explanations. Same score, a third of the work.
    scored = [generate_decision_report(sid, with_explanations=False) for sid in candidates]
    shortlist = sorted(scored, key=lambda r: r["risk_score"], reverse=True)[:top_n]
    ranked = [generate_decision_report(r["store_id"]) for r in shortlist]
    ranked.sort(key=lambda r: r["risk_score"], reverse=True)

    without_forecast = [r["store_id"] for r in ranked if not r.get("forecast_available")]

    total_stores = 0
    try:
        total_stores = int(database.get_dataset_bounds().get("total_stores") or 0)
    except Exception:  # pragma: no cover - bounds are advisory here
        total_stores = 0

    return {
        "ranked_stores": ranked,
        "summary": _fleet_risk_summary(ranked, without_forecast),
        "screened_stores": len(candidates),
        "scored_stores": len(scored),
        "fleet_size": total_stores,
        "stores_without_forecast": without_forecast,
        "methodology": (
            f"All {total_stores or 'available'} stores were ranked by recent sales trend in one "
            f"pass; the weakest {len(candidates)} were then scored with the standard risk "
            f"methodology (sales trend, forecast weakness vs the store's own average, and days "
            f"since the last promotion)."
        ),
    }


def _fleet_risk_summary(ranked: list[dict], without_forecast: list[int]) -> str:
    if not ranked:
        return "No store met the threshold for a risk ranking."

    top = ranked[0]
    if top["urgency"] == "act_now":
        lead = f"Store {top['store_id']} is the most at-risk store ({top['risk_level']}, score {top['risk_score']})."
    elif top["urgency"] == "monitor":
        lead = (f"Store {top['store_id']} leads the risk ranking ({top['risk_level']}, score "
                f"{top['risk_score']}) and is the one to watch, but nothing here needs emergency action.")
    else:
        lead = (f"These are the fleet's weakest stores *relative to each other* — but none of them "
                f"is above LOW risk on the absolute scale, so nothing here needs action this week. "
                f"Store {top['store_id']} ranks first (score {top['risk_score']}).")

    if without_forecast:
        listed = ", ".join(f"Store {s}" for s in without_forecast)
        lead += (f" {listed} {'have' if len(without_forecast) > 1 else 'has'} no forecast in this "
                 f"window, so {'their' if len(without_forecast) > 1 else 'its'} score uses sales "
                 f"trend and promo signals only — no forecast figure was estimated.")
    return lead

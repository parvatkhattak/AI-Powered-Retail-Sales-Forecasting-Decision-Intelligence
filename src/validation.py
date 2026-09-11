"""
src/validation.py
Owner: Saumya — Agentic AI Engineer

Checks a structured question (from `query_understanding.py`) against what the
data can actually support, *before* any tool runs.

This is the layer that was missing. The agent had no idea whether Store 9999
existed, whether 17 August 2025 was inside the dataset, or whether the user's
"sales increased 83% yesterday" was even true — so every one of those questions
was answered with a generic performance summary, which reads as a confident
answer to a question that was never asked.

Four things get checked here:

1. **Entity existence** — is this a real store in this dataset?
2. **Temporal coverage** — is this date inside the history, inside the
   forecast window, or outside both?
3. **Capability** — is this an operation the assistant performs at all
   (raw record dumps are not)?
4. **Premise** — if the user asserted a number, does the data agree?

A blocking finding means the question cannot be answered as asked, and the
answer has to say so rather than substitute a different question's answer.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FORECAST_DAYS
from src import database
from src.query_understanding import (
    MAX_STORES_PER_QUERY,
    Claim,
    DateRef,
    IntentSpec,
    Understanding,
)

logger = logging.getLogger(__name__)

# Used only if the database is unreachable — the agent still has to be able to
# say "that store doesn't exist" rather than crash.
_FALLBACK_STORE_RANGE = (1, 1115)


@dataclass(frozen=True)
class Coverage:
    """The boundaries of what this deployment can answer, read from the data."""
    history_start: date
    history_end: date
    forecast_start: date
    forecast_end: date
    store_ids: frozenset[int]
    total_stores: int
    forecast_days: int

    def has_store(self, store_id: int) -> bool:
        if self.store_ids:
            return store_id in self.store_ids
        low, high = _FALLBACK_STORE_RANGE
        return low <= store_id <= high

    def store_range_label(self) -> str:
        if self.store_ids:
            return f"{min(self.store_ids)}–{max(self.store_ids)}"
        return f"{_FALLBACK_STORE_RANGE[0]}–{_FALLBACK_STORE_RANGE[1]}"

    def covers_history(self, when: date) -> bool:
        return self.history_start <= when <= self.history_end

    def covers_forecast(self, when: date) -> bool:
        return self.forecast_start <= when <= self.forecast_end

    def as_dict(self) -> dict:
        """Compact form handed to the LLM, so it can describe the boundaries
        without being told them in the prompt text."""
        return {
            "history_start": self.history_start.isoformat(),
            "history_end": self.history_end.isoformat(),
            "forecast_start": self.forecast_start.isoformat(),
            "forecast_end": self.forecast_end.isoformat(),
            "forecast_days": self.forecast_days,
            "total_stores": self.total_stores,
            "store_id_range": self.store_range_label(),
        }


_coverage_cache: Coverage | None = None


def _parse_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def get_coverage(refresh: bool = False) -> Coverage:
    """Dataset boundaries, read once and cached.

    Every date and store check in the agent resolves through here, so the
    limits always come from the data that is actually loaded rather than a
    constant written into the agent.
    """
    global _coverage_cache
    if _coverage_cache is not None and not refresh:
        return _coverage_cache

    try:
        bounds = database.get_dataset_bounds()
    except Exception as exc:
        logger.warning("could not read dataset bounds, using fallback coverage: %s", exc)
        bounds = {}

    history_start = _parse_iso(bounds.get("min_date")) or date(2013, 1, 1)
    history_end = _parse_iso(bounds.get("max_date")) or date(2015, 7, 31)

    _coverage_cache = Coverage(
        history_start=history_start,
        history_end=history_end,
        # The model forecasts forward from the last day on record, so the
        # supported window is derived, never hardcoded.
        forecast_start=history_end + timedelta(days=1),
        forecast_end=history_end + timedelta(days=FORECAST_DAYS),
        store_ids=frozenset(bounds.get("store_ids") or []),
        total_stores=int(bounds.get("total_stores") or 0),
        forecast_days=FORECAST_DAYS,
    )
    return _coverage_cache


def reset_coverage_cache() -> None:
    """Used by tests that swap the database out from under the agent."""
    global _coverage_cache
    _coverage_cache = None


# ── Findings ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Finding:
    """Something the answer must state. `block` means the question as asked
    cannot be answered; `notice` means it can, with a caveat attached."""
    code: str
    severity: str      # "block" | "notice"
    message: str

    @property
    def blocking(self) -> bool:
        return self.severity == "block"


@dataclass
class ClaimVerdict:
    """The result of testing a premise the user asserted."""
    claim: Claim
    status: str            # "contradicted" | "supported" | "unverifiable"
    message: str
    actual_value: float | None = None


@dataclass
class ValidationResult:
    findings: list[Finding] = field(default_factory=list)
    known_stores: list[int] = field(default_factory=list)
    unknown_stores: list[int] = field(default_factory=list)
    plan: list[IntentSpec] = field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    coverage: Coverage | None = None

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    @property
    def notices(self) -> list[Finding]:
        return [f for f in self.findings if not f.blocking]

    @property
    def is_answerable(self) -> bool:
        """True when there is still real work to do. A blocking finding with
        no executable plan left means the honest answer is the refusal."""
        return bool(self.plan)

    def as_dict(self) -> dict:
        return {
            "blocking": [{"code": f.code, "message": f.message} for f in self.blocking_findings],
            "notices": [{"code": f.code, "message": f.message} for f in self.notices],
            "known_stores": self.known_stores,
            "unknown_stores": self.unknown_stores,
            "claims_checked": [
                {"claim": v.claim.raw, "status": v.status, "message": v.message,
                 "actual_value": v.actual_value}
                for v in self.claim_verdicts
            ],
            "coverage": self.coverage.as_dict() if self.coverage else {},
        }


# ── Individual checks ────────────────────────────────────────────────────────

def _describe_date(ref: DateRef) -> str:
    if ref.is_single_day:
        return ref.start.isoformat()
    return f"{ref.start.isoformat()} to {ref.end.isoformat()}"


def check_stores(store_candidates: list[int], coverage: Coverage) -> tuple[list[int], list[int], list[Finding]]:
    """Split the mentioned stores into ones that exist and ones that don't."""
    known = [s for s in store_candidates if coverage.has_store(s)]
    unknown = [s for s in store_candidates if not coverage.has_store(s)]

    findings: list[Finding] = []
    if unknown:
        listed = ", ".join(f"Store {s}" for s in unknown)
        is_plural = len(unknown) > 1
        findings.append(Finding(
            code="unknown_store",
            severity="block" if not known else "notice",
            message=(
                f"{listed} {'are' if is_plural else 'is'} not present in the available dataset. "
                f"This dataset covers {coverage.total_stores or 'the'} stores, "
                f"numbered {coverage.store_range_label()}. "
                f"I can't confirm or report any figure for "
                f"{'those stores' if is_plural else 'that store'}."
            ),
        ))
    return known, unknown, findings


def check_dates(dates: list[DateRef], coverage: Coverage, wants_forecast: bool) -> list[Finding]:
    """Whether each date the user named is inside history or the forecast window."""
    findings: list[Finding] = []
    for ref in dates:
        in_history = coverage.covers_history(ref.start) or coverage.covers_history(ref.end)
        in_forecast = coverage.covers_forecast(ref.start) or coverage.covers_forecast(ref.end)
        if in_history or in_forecast:
            continue

        after_end = ref.start > coverage.forecast_end
        if after_end:
            message = (
                f"{_describe_date(ref)} is outside what I can answer for. "
                f"The sales history runs {coverage.history_start.isoformat()} to "
                f"{coverage.history_end.isoformat()}, and the model forecasts only "
                f"{coverage.forecast_days} days beyond that "
                f"({coverage.forecast_start.isoformat()} to {coverage.forecast_end.isoformat()}). "
                f"I have no data or forecast for that date and won't estimate one."
            )
            code = "date_beyond_horizon"
        else:
            message = (
                f"{_describe_date(ref)} is before this dataset begins "
                f"({coverage.history_start.isoformat()}). I have no records for it."
            )
            code = "date_before_history"

        findings.append(Finding(code=code, severity="block", message=message))
    return findings


def check_horizon(horizon_days: int | None, coverage: Coverage) -> list[Finding]:
    """A request to look further ahead than the model was built to see."""
    if horizon_days is None or horizon_days <= coverage.forecast_days:
        return []
    return [Finding(
        code="horizon_too_long",
        severity="notice",
        message=(
            f"You asked about the next {horizon_days} days, but the forecast model "
            f"produces {coverage.forecast_days} days "
            f"({coverage.forecast_start.isoformat()} to {coverage.forecast_end.isoformat()}). "
            f"I'll answer for that window and nothing beyond it."
        ),
    )]


def check_operation(understanding: Understanding) -> list[Finding]:
    """Operations the assistant does not perform, whatever the phrasing.

    A raw-record request is refused rather than quietly answered with a
    summary: silently substituting a different operation is worse than saying
    no, because the user believes they got what they asked for.
    """
    if not understanding.entities.wants_raw_records:
        return []
    return [Finding(
        code="raw_records_refused",
        severity="block",
        message=(
            "I don't return raw record-level data or full table exports — this "
            "assistant reports aggregated metrics only, and the dataset behind it "
            "is hundreds of thousands of rows. I'm not going to substitute a "
            "summary and present it as what you asked for. What I can give you "
            "instead: per-store daily averages and trends, promo uplift rankings, "
            "fleet-wide top/bottom performers, or a 7-day forecast for named stores."
        ),
    )]


def check_reference_resolution(understanding: Understanding, context: dict) -> list[Finding]:
    """A follow-up that points at something the conversation never established."""
    entities = understanding.entities
    if not entities.references_prior_context or entities.store_candidates:
        return []
    if context.get("previous_stores"):
        return []
    return [Finding(
        code="unresolved_reference",
        severity="block",
        message=(
            "Your question refers back to something we haven't discussed yet in "
            "this conversation, so I don't know which stores you mean. Name the "
            "stores and I'll answer — for example \"which of Stores 125 and 220 "
            "should I prioritise?\"."
        ),
    )]


def check_store_cap(understanding: Understanding) -> list[Finding]:
    if not understanding.entities.exceeds_store_cap:
        return []
    return [Finding(
        code="too_many_stores",
        severity="notice",
        message=(
            f"You asked about more stores than I analyse in one answer. I'll cover "
            f"the first {MAX_STORES_PER_QUERY} — ask again with a shorter list for the rest."
        ),
    )]


# ── Premise checking ─────────────────────────────────────────────────────────

def _article(value: float) -> str:
    """"an 83% increase", not "a 83% increase" — the article follows how the
    number is read aloud, which is 8/11/18 and their multiples."""
    spoken = f"{value:g}"
    return "an" if spoken[0] == "8" or spoken.startswith(("11", "18")) else "a"


def _pct_change(earlier: float, later: float) -> float | None:
    if not earlier:
        return None
    return (later - earlier) / earlier * 100.0


def verify_claims(claims: list[Claim], store_ids: list[int], dates: list[DateRef],
                  coverage: Coverage) -> list[ClaimVerdict]:
    """Test each asserted number against the real series.

    Without this, "why did Store 125's sales increase 83% yesterday?" was
    answered as a "why" question — the agent explained a rise that the data
    shows never happened. The premise has to be checked before the explanation
    is attempted.
    """
    verdicts: list[ClaimVerdict] = []
    if not claims:
        return verdicts

    if not store_ids:
        return [ClaimVerdict(c, "unverifiable",
                             "No store was named, so I can't check that figure against the data.")
                for c in claims]

    store_id = store_ids[0]
    try:
        metrics = database.get_store_metrics([store_id], days=60)
    except Exception as exc:
        logger.warning("claim verification could not load metrics for store %s: %s", store_id, exc)
        return [ClaimVerdict(c, "unverifiable",
                             "I couldn't load that store's sales history to check the figure.")
                for c in claims]

    if metrics is None or metrics.empty:
        return [ClaimVerdict(c, "unverifiable",
                             f"No sales history is available for Store {store_id} to check that figure against.")
                for c in claims]

    rows = metrics[metrics["store_id"] == store_id] if "store_id" in metrics.columns else metrics
    rows = rows.sort_values("date")
    # Closed days are real zeros in this dataset; comparing against them would
    # manufacture a -100% "change" that says nothing about performance.
    trading = rows[rows["sales"] > 0]
    if trading.empty:
        return [ClaimVerdict(c, "unverifiable",
                             f"Store {store_id} has no trading days on record in the window I check.")
                for c in claims]

    target_day = next((d.start for d in dates if d.is_single_day), None)
    day_rows = trading[trading["date"].dt.date <= target_day] if target_day else trading
    if day_rows.empty:
        day_rows = trading

    latest = day_rows.iloc[-1]
    previous = day_rows.iloc[-2] if len(day_rows) >= 2 else None

    day_change = _pct_change(float(previous["sales"]), float(latest["sales"])) if previous is not None else None
    latest_date = latest["date"].date()
    latest_sales = float(latest["sales"])

    for claim in claims:
        if claim.kind == "level":
            gap = abs(latest_sales - claim.value)
            close_enough = gap <= max(1.0, 0.02 * max(claim.value, 1.0))
            verdicts.append(ClaimVerdict(
                claim,
                "supported" if close_enough else "contradicted",
                (f"Store {store_id} recorded €{latest_sales:,.0f} on {latest_date.isoformat()}, "
                 f"not {claim.value:,.0f}.") if not close_enough else
                (f"Confirmed: Store {store_id} recorded €{latest_sales:,.0f} on {latest_date.isoformat()}."),
                actual_value=round(latest_sales, 2),
            ))
            continue

        if day_change is None:
            verdicts.append(ClaimVerdict(
                claim, "unverifiable",
                f"Store {store_id} has only one trading day on record in that window, "
                f"so there is nothing to compare it against.",
            ))
            continue

        claimed = claim.value if claim.direction == "up" else -claim.value
        same_direction = (claimed >= 0) == (day_change >= 0)
        # Tolerance scales with the claim: 2 points at minimum, 20% of the
        # claimed magnitude for larger ones.
        tolerance = max(2.0, abs(claimed) * 0.2)
        matches = same_direction and abs(day_change - claimed) <= tolerance

        previous_date = previous["date"].date()
        direction_word = "up" if day_change >= 0 else "down"
        detail = (
            f"Store {store_id} recorded €{latest_sales:,.0f} on {latest_date.isoformat()} "
            f"versus €{float(previous['sales']):,.0f} on the previous trading day "
            f"({previous_date.isoformat()}) — {direction_word} {abs(day_change):.1f}%."
        )
        verdicts.append(ClaimVerdict(
            claim,
            "supported" if matches else "contradicted",
            (f"Confirmed. {detail}" if matches else
             f"I can't verify {_article(claim.value)} {claim.value:g}% "
             f"{'increase' if claim.direction == 'up' else 'decrease'}. {detail}"),
            actual_value=round(day_change, 2),
        ))

    return verdicts


# ── Plan building ────────────────────────────────────────────────────────────

_NEEDS_A_STORE = ("forecast", "whatif")


def build_plan(understanding: Understanding, known_stores: list[int],
               blocking_codes: set[str], coverage: "Coverage | None" = None) -> list[IntentSpec]:
    """The executable subset of what was asked.

    Steps whose stores all turned out to be unknown are dropped rather than
    silently re-pointed at some other store, and an intent that structurally
    needs a store ("forecast Store 9999") is dropped when none survives.
    """
    plan: list[IntentSpec] = []
    for spec in understanding.intents:
        # A raw-record request is a refusal, never a step: the finding says so,
        # and the *other* parts of a stacked question still get answered.
        if spec.type in ("out_of_scope", "raw_data"):
            continue
        stores = [s for s in spec.stores if s in known_stores]
        if spec.stores and not stores:
            continue  # every store in this step was unknown
        if not stores and spec.type in _NEEDS_A_STORE:
            continue
        if not stores and "unresolved_reference" in blocking_codes:
            continue
        # Scoped per step, so "what is its 7-day forecast, and what will sales
        # be in 2027" keeps the first and drops only the second.
        if coverage is not None and any(
            not (coverage.covers_history(d.start) or coverage.covers_history(d.end)
                 or coverage.covers_forecast(d.start) or coverage.covers_forecast(d.end))
            for d in spec.dates
        ):
            continue
        plan.append(IntentSpec(
            type=spec.type,
            stores=stores,
            scope="store" if stores else "fleet",
            clause=spec.clause,
            metrics=list(spec.metrics),
            dates=list(spec.dates),
        ))

    deduped: list[IntentSpec] = []
    for spec in plan:
        twin = next((d for d in deduped
                     if d.type == spec.type and d.stores == spec.stores), None)
        if twin is not None:
            twin.metrics = list(dict.fromkeys(twin.metrics + spec.metrics))
            continue
        deduped.append(spec)
    return deduped


def validate(understanding: Understanding, context: dict | None = None,
             check_premises: bool = True) -> ValidationResult:
    """Run every check and return what the answer is allowed to claim."""
    context = context or {}
    coverage = get_coverage()

    known, unknown, findings = check_stores(understanding.entities.store_candidates, coverage)

    wants_forecast = any(i.type in ("forecast", "whatif") for i in understanding.intents)
    findings += check_dates(understanding.entities.dates, coverage, wants_forecast)
    findings += check_horizon(understanding.entities.horizon_days, coverage)
    findings += check_operation(understanding)
    findings += check_reference_resolution(understanding, context)
    findings += check_store_cap(understanding)

    verdicts: list[ClaimVerdict] = []
    if check_premises and understanding.entities.claims and known:
        verdicts = verify_claims(understanding.entities.claims, known,
                                 understanding.entities.dates, coverage)
        for verdict in verdicts:
            # Both directions are surfaced: a user who asserts something true
            # should see it confirmed against the data, not left wondering
            # whether it was checked at all.
            if verdict.status in ("contradicted", "supported"):
                findings.append(Finding(
                    code="false_premise" if verdict.status == "contradicted" else "premise_confirmed",
                    severity="notice",
                    message=verdict.message,
                ))

    blocking_codes = {f.code for f in findings if f.blocking}
    # A blocking finding removes the part of the question it applies to, not
    # necessarily the whole thing. If nothing executable is left, the refusal
    # *is* the answer; if something is, the answer states the refusal and then
    # answers the rest.
    plan = build_plan(understanding, known, blocking_codes, coverage)

    return ValidationResult(
        findings=findings,
        known_stores=known,
        unknown_stores=unknown,
        plan=plan,
        claim_verdicts=verdicts,
        coverage=coverage,
    )

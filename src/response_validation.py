"""
src/response_validation.py
Owner: Saumya — Agentic AI Engineer

The last gate before text reaches the user.

Everything upstream decides what the answer *should* contain. This module
checks what it actually contains, and it is the only place that can see both
at once. A system prompt asking the model not to invent numbers is a request;
this is the check.

Ten named checks run on every answer. The important property is that they are
mechanical: each one compares the response text against the grounded data the
pipeline collected, so a failure is a fact about the text, not a judgement.

When the LLM-composed answer fails a check, the agent falls back to the
deterministic composition — which is grounded by construction, because it can
only print numbers it registered — rather than shipping the failing text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

# Numbers a store manager would read as facts about the business. Plain
# integers are handled separately (see _scan_numbers) because ordinals and
# list markers are not claims about anything.
_CURRENCY_RE = re.compile(r"[€$£₹]\s*(-?[\d,]+(?:\.\d+)?)")
_PERCENT_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)\s*%")
_STORE_RE = re.compile(r"\bstores?\s+(\d{1,5})\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ORDINAL_LINE_RE = re.compile(r"^\s*\d{1,3}[.)]\s+", re.MULTILINE)

_LEAK_MARKERS = (
    "traceback", "sqlalchemy", "sqlite3.", "sqlite://", "no such table",
    "select * from", "drop table", "openrouter_api_key", "sk-or-", "/home/",
    "site-packages", "keyerror", "attributeerror", 'file "',
)

# A priority and a "nothing to do" cannot both be true of the same store.
_NO_ACTION_PHRASES = ("no action needed", "no urgent action", "is stable", "nothing needs doing")
_PRIORITY_PHRASES = ("priority:", "act this week", "needs attention first", "urgent action required")

_SECTION_HINTS = {
    "performance": ("/day", "average", "trend", "uplift", "performing", "sales"),
    "forecast": ("forecast", "predicted", "next week", "expected"),
    "recommend": ("recommendation", "priority", "risk", "watch", "no action"),
    "whatif": ("with the promotion", "without", "difference", "scenario", "what-if"),
}


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


class Grounding:
    """The set of values an answer is allowed to state.

    Populated from the tool results, and added to by the deterministic
    composer as it computes each figure it prints. A number that is in neither
    place was invented somewhere, which is exactly what we want to catch.
    """

    def __init__(self) -> None:
        self.numbers: set[float] = set()
        self.store_ids: set[int] = set()
        self.dates: set[str] = set()

    def add_number(self, value) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
            return
        self.numbers.add(number)
        # The same figure gets printed rounded, so the rounded forms are
        # grounded too — €8,333 and 8333.47 are the same claim.
        self.numbers.update({round(number), round(number, 1), round(number, 2), abs(number)})

    def add_store(self, store_id) -> None:
        try:
            self.store_ids.add(int(store_id))
        except (TypeError, ValueError):
            return

    def add_date(self, value) -> None:
        if isinstance(value, date):
            self.dates.add(value.isoformat())
        elif isinstance(value, str) and _ISO_DATE_RE.search(value):
            self.dates.add(_ISO_DATE_RE.search(value).group(0))

    def ingest(self, obj, depth: int = 0) -> None:
        """Walk a tool-results structure and ground everything numeric in it."""
        if depth > 8:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, (int, str)) and str(key).isdigit():
                    self.add_store(key)
                if isinstance(key, str) and "store" in key.lower():
                    self.add_store(value)
                if isinstance(key, str) and "date" in key.lower():
                    self.add_date(value)
                self.ingest(value, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                self.ingest(item, depth + 1)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float)):
            self.add_number(obj)
        elif isinstance(obj, str):
            self.add_date(obj)
            match = _ISO_DATE_RE.search(obj)
            if not match:
                # Thousands separators included: a figure already formatted as
                # "€9,344" inside an observation string is one number, not two.
                for token in re.findall(r"-?\d[\d,]*(?:\.\d+)?", obj):
                    self.add_number(token.replace(",", ""))

    def allows_number(self, value: float) -> bool:
        for grounded in self.numbers:
            if abs(grounded - value) <= max(0.51, abs(grounded) * 0.005):
                return True
        return False


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures) or "all checks passed"


def _scan_numbers(response: str) -> list[float]:
    """Currency amounts and percentages stated in the answer.

    Store IDs, ISO dates and markdown list markers are removed first — they're
    identifiers and structure, not claims about the business.
    """
    text = _ORDINAL_LINE_RE.sub("", response)
    text = _STORE_RE.sub(" ", text)
    text = _ISO_DATE_RE.sub(" ", text)

    values = []
    for regex in (_CURRENCY_RE, _PERCENT_RE):
        for match in regex.finditer(text):
            number = _to_float(match.group(1))
            if number is not None:
                values.append(number)
    return values


# ── The ten checks ───────────────────────────────────────────────────────────

def check_not_empty(response: str) -> CheckResult:
    ok = bool(response and len(response.strip()) >= 20)
    return CheckResult("not_empty", ok, "" if ok else "response is empty or too short to be an answer")


def check_no_internal_leakage(response: str) -> CheckResult:
    lowered = response.lower()
    leaked = [marker for marker in _LEAK_MARKERS if marker in lowered]
    return CheckResult("no_internal_leakage", not leaked,
                       f"response exposes internals: {leaked}" if leaked else "")


def check_numeric_grounding(response: str, grounding: Grounding) -> CheckResult:
    ungrounded = [v for v in _scan_numbers(response) if not grounding.allows_number(v)]
    return CheckResult("numeric_grounding", not ungrounded,
                       f"figures not present in the tool results: {ungrounded[:6]}" if ungrounded else "")


def check_store_grounding(response: str, grounding: Grounding) -> CheckResult:
    mentioned = {int(m.group(1)) for m in _STORE_RE.finditer(response)}
    if not grounding.store_ids:
        return CheckResult("store_grounding", True, "")
    unknown = sorted(mentioned - grounding.store_ids)
    return CheckResult("store_grounding", not unknown,
                       f"answer names stores that were never looked up: {unknown}" if unknown else "")


def check_date_grounding(response: str, grounding: Grounding, coverage: dict | None) -> CheckResult:
    """No date outside the dataset may be stated as if it had data behind it."""
    if not coverage:
        return CheckResult("date_grounding", True, "")
    start, end = coverage.get("history_start"), coverage.get("forecast_end")
    if not start or not end:
        return CheckResult("date_grounding", True, "")

    out_of_range = []
    for match in _ISO_DATE_RE.finditer(response):
        stated = match.group(0)
        if stated in grounding.dates:
            continue
        if stated < start or stated > end:
            out_of_range.append(stated)
    return CheckResult("date_grounding", not out_of_range,
                       f"answer states dates outside dataset coverage: {out_of_range}" if out_of_range else "")


def check_notices_present(response: str, required_notices: list[str]) -> CheckResult:
    """A validation finding the user has to see must actually be in the text.

    This is the check that stops the old behaviour — a question about Store
    9999 being answered with a fleet summary that never mentions the store
    doesn't exist.
    """
    lowered = response.lower()
    missing = []
    for notice in required_notices:
        # Match on the notice's distinguishing words rather than the whole
        # sentence, so an LLM rephrasing it still counts as having said it.
        keywords = [w for w in re.findall(r"[a-z0-9]{4,}", notice.lower())][:6]
        if keywords and sum(1 for w in keywords if w in lowered) < max(2, len(keywords) // 2):
            missing.append(notice[:60])
    return CheckResult("required_notices", not missing,
                       f"answer omits what the user has to be told: {missing}" if missing else "")


def check_no_self_contradiction(response: str) -> CheckResult:
    lowered = response.lower()
    # Per store, not per response: a five-store ranking legitimately contains
    # one store to act on and four with nothing to do.
    for block in re.split(r"\n(?=\s*\d{1,2}[.)]\s)|\n\n", response):
        chunk = block.lower()
        if any(p in chunk for p in _PRIORITY_PHRASES) and any(p in chunk for p in _NO_ACTION_PHRASES):
            return CheckResult("no_self_contradiction", False,
                               f"one store is called both a priority and stable: {block.strip()[:120]}")
    if "no action needed for any" in lowered and "priority:" in lowered:
        return CheckResult("no_self_contradiction", False,
                           "summary names a priority while saying no action is needed")
    return CheckResult("no_self_contradiction", True, "")


def check_covers_all_intents(response: str, intent_types: list[str]) -> CheckResult:
    """A stacked question must not be answered with one of its parts."""
    lowered = response.lower()
    unanswered = []
    for intent in dict.fromkeys(intent_types):
        hints = _SECTION_HINTS.get(intent)
        if hints and not any(h in lowered for h in hints):
            unanswered.append(intent)
    return CheckResult("covers_all_intents", not unanswered,
                       f"question asked for {unanswered} and the answer never addresses it" if unanswered else "")


def check_has_sources(response: str, expected_sources: list[str]) -> CheckResult:
    if not expected_sources:
        return CheckResult("has_sources", True, "")
    ok = "sources" in response.lower()
    return CheckResult("has_sources", ok, "" if ok else "answer cites no data sources")


def check_no_unchecked_premise_echo(response: str, contradicted: list[str]) -> CheckResult:
    """If the user asserted a figure the data contradicts, the answer must not
    repeat it as if it were true."""
    if not contradicted:
        return CheckResult("premise_corrected", True, "")
    lowered = response.lower()
    corrective = ("can't verify", "cannot verify", "not verify", "no evidence", "actually",
                  "instead", "does not show", "doesn't show", "not present", "rather than")
    ok = any(c in lowered for c in corrective)
    return CheckResult("premise_corrected", ok,
                       "" if ok else "answer repeats a claim the data contradicts without correcting it")


def validate_response(
    response: str,
    grounding: Grounding,
    intent_types: list[str] | None = None,
    required_notices: list[str] | None = None,
    expected_sources: list[str] | None = None,
    contradicted_claims: list[str] | None = None,
    coverage: dict | None = None,
) -> ValidationReport:
    """Run all ten checks and report which failed."""
    return ValidationReport(checks=[
        check_not_empty(response),
        check_no_internal_leakage(response),
        check_numeric_grounding(response, grounding),
        check_store_grounding(response, grounding),
        check_date_grounding(response, grounding, coverage),
        check_notices_present(response, required_notices or []),
        check_no_self_contradiction(response),
        check_covers_all_intents(response, intent_types or []),
        check_has_sources(response, expected_sources or []),
        check_no_unchecked_premise_echo(response, contradicted_claims or []),
    ])

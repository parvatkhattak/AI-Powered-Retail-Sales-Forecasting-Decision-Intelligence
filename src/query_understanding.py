"""
src/query_understanding.py
Owner: Saumya — Agentic AI Engineer

Turns raw user text into a structured, checkable representation *before* any
tool runs.

Why this exists: the agent used to go straight from a raw string to an intent
label, so everything the question actually said — the date, the store number,
the claim the user asserted, the fact that it was six questions stacked into
one — was invisible to the rest of the pipeline. A question about 2025 looked
identical to a question about last week, and a question about Store 9999
looked identical to one about Store 125.

Nothing in this module touches the database, the model or the LLM. It is pure,
deterministic text -> structure, which is what makes it testable and what lets
`validation.py` check the structure against what the data can actually support.

Pipeline position:

    normalize -> split into clauses -> per-clause intent -> entities -> (validation.py)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta

# A chat answer comparing more than a handful of stores is unreadable, and each
# store costs a full forecast + SHAP run.
MAX_STORES_PER_QUERY = 5

INTENT_TYPES = ("performance", "forecast", "recommend", "whatif", "raw_data", "out_of_scope")


# ── Normalization ────────────────────────────────────────────────────────────

# Written as (pattern, replacement) so the rewrite is visible and testable
# rather than buried in a chain of .replace() calls.
_CONTRACTIONS = [
    (r"\bwhat'?s\b", "what is"),
    (r"\bwhere'?s\b", "where is"),
    (r"\bhow'?s\b", "how is"),
    (r"\bwho'?s\b", "who is"),
    # An apostrophe is required wherever the bare spelling is a real word of
    # its own — expanding "its" to "it is" mangled "what is its forecast?".
    (r"\bit's\b", "it is"),
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bwe're\b", "we are"),
    (r"\bthey're\b", "they are"),
    (r"\blet's\b", "let us"),
    (r"\bdon'?t\b", "do not"),
    (r"\bdoesn'?t\b", "does not"),
    (r"\bdidn'?t\b", "did not"),
    (r"\bi'?m\b", "i am"),
    (r"\bi'?ve\b", "i have"),
]

# Shorthand a store manager actually types. Expanded so the keyword layers
# below only ever have to recognise one spelling of each concept.
_ABBREVIATIONS = [
    (r"\bpls\b|\bplz\b", "please"),
    (r"\bthx\b|\bty\b", "thanks"),
    (r"\bnxt\b", "next"),
    (r"\bwk\b", "week"),
    (r"\bwks\b", "weeks"),
    (r"\bmth\b|\bmnth\b", "month"),
    (r"\byday\b|\by'?day\b", "yesterday"),
    (r"\btmrw\b|\btmw\b", "tomorrow"),
    (r"\bperf\b", "performance"),
    (r"\bfcst\b", "forecast"),
    (r"\bavg\b", "average"),
    (r"\brev\b", "revenue"),
    (r"\bst\s+(\d+)\b", r"store \1"),
    (r"\bstr\s*#?\s*(\d+)\b", r"store \1"),
    (r"\bstore\s*no\.?\s*(\d+)\b", r"store \1"),
    (r"\bstore\s*#\s*(\d+)\b", r"store \1"),
    (r"\bwhat\s*if\b", "what if"),
]

_DASHES = dict.fromkeys(map(ord, "–—−‒―"), "-")
_QUOTES = {ord("’"): "'", ord("‘"): "'", ord("“"): '"', ord("”"): '"'}


def normalize(query: str) -> str:
    """Canonical form used by every downstream matcher.

    Unicode-normalised, unified dashes/quotes/currency, expanded contractions
    and shorthand, collapsed whitespace. Lowercased — nothing downstream needs
    the original casing, and case-insensitive regexes everywhere were how
    "Store 125" and "store 125" drifted apart in the first place.
    """
    text = unicodedata.normalize("NFKC", query or "")
    text = text.translate(_DASHES).translate(_QUOTES)
    text = text.lower()
    for pattern, replacement in _CONTRACTIONS + _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    # Indian digit grouping (8,72,431) and Western (872,431) both become plain
    # digits so a claimed figure can be compared numerically.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Clause splitting (multi-intent) ──────────────────────────────────────────

# A stacked question is several questions in one string. Splitting on sentence
# and coordination boundaries is what lets each part get its own intent instead
# of the whole thing collapsing to whichever keyword happened to match first.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[?!])\s+"                       # end of a question
    r"|\s*[;\n]+\s*"                      # explicit separators
    r"|(?<=[a-z0-9%)])\.\s+(?=[a-z])"     # sentence break
    r"|\s+(?:and\s+)?also\s+"             # "and also ..."
    r"|\s*,\s*(?:and\s+|then\s+)?(?=(?:what|which|how|why|when|where|who|should\s+i|can\s+i|do\s+i|tell\s+me|show\s+me|give\s+me|list|compare|rank)\b)"
    r"|\s+and\s+(?=(?:what|which|how|why|when|where|who|should\s+i|can\s+i|do\s+i|tell\s+me|show\s+me|give\s+me)\b)",
    re.IGNORECASE,
)

# "1) ... 2) ... 3) ..." style enumerations.
_ENUMERATION_RE = re.compile(r"(?:(?<=\s)|^)\d{1,2}\s*[.)]\s+")

_MIN_CLAUSE_WORDS = 2

# A clause that doesn't open like a question or an instruction is a qualifier,
# not a separate ask: "based on historical performance and your forecast" sets
# up the question that follows it rather than being one.
_CLAUSE_HEAD_RE = re.compile(
    r"^(?:what|which|why|how|when|where|who|whose|is|are|was|were|do|does|did|can|could"
    r"|should|would|will|shall|tell|show|give|list|compare|rank|find|explain|analy[sz]e"
    r"|summari[sz]e|predict|forecast|recommend|suggest|help|any)\b",
    re.IGNORECASE,
)


def split_clauses(normalized_query: str) -> list[str]:
    """The question broken into the separate things it actually asks."""
    if not normalized_query:
        return []

    parts: list[str] = []
    for enumerated in _ENUMERATION_RE.split(normalized_query):
        parts.extend(_CLAUSE_SPLIT_RE.split(enumerated))

    clauses = []
    for part in parts:
        cleaned = (part or "").strip(" ,.;-")
        if not cleaned:
            continue
        if len(cleaned.split()) < _MIN_CLAUSE_WORDS and clauses:
            # A dangling fragment ("and why") belongs to the clause before it.
            clauses[-1] = f"{clauses[-1]} {cleaned}"
            continue
        clauses.append(cleaned)

    return _merge_qualifiers(clauses) or [normalized_query]


def _merge_qualifiers(clauses: list[str]) -> list[str]:
    """Attach each non-question fragment to the question it belongs to."""
    if len(clauses) < 2:
        return clauses

    merged: list[str] = []
    pending: list[str] = []
    for clause in clauses:
        if _CLAUSE_HEAD_RE.match(clause) or clause.endswith("?"):
            merged.append(" ".join(pending + [clause]))
            pending = []
        else:
            pending.append(clause)

    if pending:
        # Trailing qualifiers have no question after them, so they belong to
        # the one before — or stand alone if there was none.
        if merged:
            merged[-1] = f"{merged[-1]} {' '.join(pending)}"
        else:
            merged.append(" ".join(pending))
    return merged


# ── Store IDs ────────────────────────────────────────────────────────────────

# Only numbers that follow the word "store(s)" count as store IDs — avoids
# false positives like "top 10 stores" or "next 7 days".
_STORE_MENTION_RE = re.compile(
    r"stores?\s*(?:id[s]?)?\s*[:#]?\s*"
    # Separators come *before* each number and repeat, so compound joins like
    # "400, and 500" are matched as one list rather than stopping at "and".
    r"((?:(?:\s*(?:,|and|&|/|-|to|through)\s*)*\d+\s*)+)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(r"(\d+)\s*(?:-|\bto\b|\bthrough\b)\s*(\d+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")

# Guards the range expander against "stores 1-999999" building a huge list
# before anything gets a chance to truncate it.
_ABSURD_STORE_ID = 1_000_000


def _expand_ranges(span: str) -> str:
    """Rewrite "100-500" / "100 to 500" into the individual IDs they stand for."""
    def expand(match: re.Match) -> str:
        low, high = int(match.group(1)), int(match.group(2))
        if low > high:
            low, high = high, low
        high = min(high, low + MAX_STORES_PER_QUERY)
        return " ".join(str(n) for n in range(low, high + 1))

    return _RANGE_RE.sub(expand, span)


def _unify_dashes(text: str) -> str:
    """Callers outside `understand()` pass raw text, where an en-dash range
    ("stores 100–500") would not match the ASCII-hyphen pattern."""
    return (text or "").translate(_DASHES)


def extract_store_candidates(query: str) -> list[int]:
    """Every number the user offered as a store ID, in the order written.

    Deliberately *not* filtered against the real store list: "Store 9999" has
    to survive this far so validation can tell the user that store doesn't
    exist, instead of the number vanishing and the question being answered as
    if no store had been named at all — which is exactly what used to happen.
    """
    ids: list[int] = []
    seen: set[int] = set()
    for mention in _STORE_MENTION_RE.finditer(_unify_dashes(query)):
        for num in _NUMBER_RE.findall(_expand_ranges(mention.group(1))):
            sid = int(num)
            if sid < _ABSURD_STORE_ID and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


# A bare number is only a store ID inside a frame that can only be about a
# store. Matching any number in range would turn "sales of 300 units" and
# "over 500 customers" into store lookups, so the frame does the work and the
# ID range only confirms it.
_BARE_STORE_FRAMES = (
    # "how is 250 <anything>" — the trailing verb is deliberately not required.
    # It used to be ("performing|doing|…"), which meant a typo like
    # "how is 250 perrforming" silently produced no store at all.
    re.compile(r"\bhow(?:\s+is|\s+are|\s+was|\s+did)?\s+#?(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\b(?:what|how)\s+about\s+#?(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\b(?:tell me about|info(?:rmation)? on|details? on|check|look at|show me|update on|status of)\s+#?(\d{1,4})\b", re.IGNORECASE),
    # Two numbers joined by or/vs/and inside a comparison — "which store is
    # better, 100 or 500?" named both stores and neither was extracted.
    re.compile(r"\b(?:compare|versus|vs\.?|between)\s+#?(\d{1,4})\s+(?:and|or|with|to|vs\.?|versus)\s+#?(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"#?(\d{1,4})\s+(?:or|vs\.?|versus)\s+#?(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"^#?(\d{1,4})\s*\??$"),
)

# A number followed by any of these is counting something, not naming a store.
_UNIT_AFTER_RE = re.compile(
    r"^\s*(?:units?|items?|products?|pieces|customers?|visitors?|transactions?|sales|euros?|dollars?"
    r"|pounds?|rupees?|days?|weeks?|months?|years?|stores?|shops?|outlets?|percent|%|rows?|records?"
    r"|orders?|people)\b",
    re.IGNORECASE,
)
# A number preceded by any of these is a rank, a count or a quantity.
_QUANTIFIER_BEFORE_RE = re.compile(
    r"\b(?:top|bottom|best|worst|first|last|next|past|previous|over|under|above|below|about|around"
    r"|approximately|roughly|more than|less than|at least|at most|[€$£₹])\s*$",
    re.IGNORECASE,
)


# Guards the loosest two frames ("N or M", and the bare "how is N") to clauses
# that are recognisably about stores at all.
_STORE_CONTEXT_RE = re.compile(
    r"\bstores?\b|\bshops?\b|\boutlets?\b|\bperform\w*\b|\bsales?\b|\brevenue\b"
    r"|\bdoing\b|\bbetter\b|\bworse\b|\bcompar\w*\b|\bprioriti[sz]\w*\b|\bforecast\w*\b"
    r"|\bpromo\w*\b|\btrend\w*\b|\brisk\w*\b|\bfocus\b",
    re.IGNORECASE,
)
# Only the bare "N or M" frame needs the context guard. "How is 250 …" is
# already specific enough on its own, and gating it on retail vocabulary meant
# a typo in that very word ("perrforming") dropped the store entirely — the
# unit-noun and ID-range guards are what actually keep quantities out.
_LOOSE_FRAME_INDEXES = (4,)


def extract_bare_store_candidates(query: str, low: int = 1, high: int = 1115) -> list[int]:
    """Store IDs named without the word "store" — "how is 300 performing?".

    Three independent conditions all have to hold: the number sits in a frame
    that can only be referring to a store, it isn't preceded by a quantifier or
    followed by a unit noun, and it falls inside the real store-ID range. The
    frame is what keeps "sales of 300 units" from becoming Store 300.
    """
    text = _unify_dashes(query)
    has_store_context = bool(_STORE_CONTEXT_RE.search(text))
    found: list[int] = []
    for index, frame in enumerate(_BARE_STORE_FRAMES):
        if index in _LOOSE_FRAME_INDEXES and not has_store_context:
            continue
        for match in frame.finditer(text):
            for group in range(1, (match.lastindex or 1) + 1):
                raw = match.group(group)
                if raw is None:
                    continue
                # Checked against what precedes the whole frame, not the
                # number: "tell me about 300" ends in "about", which is itself
                # on the quantifier list and would otherwise veto its own frame.
                if _QUANTIFIER_BEFORE_RE.search(text[:match.start()]):
                    continue
                if _UNIT_AFTER_RE.match(text[match.end(group):]):
                    continue
                value = int(raw)
                if low <= value <= high and value not in found:
                    found.append(value)
    return found


def mentions_more_stores_than(query: str, cap: int = MAX_STORES_PER_QUERY) -> bool:
    """True when the query asked about more stores than the cap allows."""
    mentioned: set[int] = set()
    for mention in _STORE_MENTION_RE.finditer(_unify_dashes(query)):
        for num in _NUMBER_RE.findall(mention.group(1)):
            sid = int(num)
            if sid < _ABSURD_STORE_ID:
                mentioned.add(sid)
        for match in _RANGE_RE.finditer(mention.group(1)):
            low, high = sorted((int(match.group(1)), int(match.group(2))))
            if high - low > cap:
                return True
    return len(mentioned) > cap


# ── Dates ────────────────────────────────────────────────────────────────────

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# Day-first: this is a German retail chain, and "17/08/2025" means 17 August.
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})[a-z]*\.?(?:,?\s+(\d{{4}}))?\b"
)
_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALT})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b"
)
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_ALT})[a-z]*\.?\s+(\d{{4}})\b")
_BARE_YEAR_RE = re.compile(r"\b(19|20)(\d{2})\b")

_HORIZON_RE = re.compile(
    r"\b(?:next|coming|following|upcoming)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)\b"
    r"|\b(?:in|over)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)(?:\s+time)?\b",
    re.IGNORECASE,
)
_LOOKBACK_RE = re.compile(
    r"\b(?:last|past|previous|trailing)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)\b",
    re.IGNORECASE,
)

_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}


@dataclass(frozen=True)
class DateRef:
    """A date or span the user referred to, resolved to concrete days.

    `basis` records how it was written, because that changes what a failed
    validation should say: an absolute date outside coverage is "we have no
    data for that day", a relative one is "'yesterday' means <dataset date>".
    """
    raw: str
    start: date
    end: date
    basis: str          # "absolute" | "relative"
    precision: str      # "day" | "month" | "year" | "span"

    @property
    def is_single_day(self) -> bool:
        return self.start == self.end


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _month_span(year: int, month: int) -> tuple[date, date] | None:
    start = _safe_date(year, month, 1)
    if start is None:
        return None
    end = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return start, end


def extract_dates(normalized_query: str, reference_date: date) -> list[DateRef]:
    """Dates in the question, resolved against `reference_date`.

    `reference_date` is the newest day the dataset actually contains, never
    today's real-world date. In a historical dataset "yesterday" can only
    sensibly mean the day before the last day on record; resolving it against
    the wall clock silently produced a date a decade past the end of the data.
    """
    found: list[DateRef] = []
    consumed: list[tuple[int, int]] = []

    def claim(match: re.Match) -> bool:
        span = match.span()
        if any(span[0] < end and start < span[1] for start, end in consumed):
            return False
        consumed.append(span)
        return True

    def add(match: re.Match, start: date, end: date, basis: str, precision: str) -> None:
        found.append(DateRef(match.group(0).strip(), start, end, basis, precision))

    for match in _ISO_DATE_RE.finditer(normalized_query):
        parsed = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if parsed and claim(match):
            add(match, parsed, parsed, "absolute", "day")

    for match in _SLASH_DATE_RE.finditer(normalized_query):
        parsed = _safe_date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        if parsed and claim(match):
            add(match, parsed, parsed, "absolute", "day")

    for regex, day_group, month_group in ((_DAY_MONTH_RE, 1, 2), (_MONTH_DAY_RE, 2, 1)):
        for match in regex.finditer(normalized_query):
            month = _MONTHS[match.group(month_group)[:4].rstrip(".")[:4]] if False else _MONTHS.get(
                match.group(month_group)
            )
            if month is None:
                continue
            year = int(match.group(3)) if match.group(3) else reference_date.year
            parsed = _safe_date(year, month, int(match.group(day_group)))
            if parsed and claim(match):
                add(match, parsed, parsed, "absolute", "day")

    for match in _MONTH_YEAR_RE.finditer(normalized_query):
        span = _month_span(int(match.group(2)), _MONTHS[match.group(1)])
        if span and claim(match):
            add(match, span[0], span[1], "absolute", "month")

    for match in _BARE_YEAR_RE.finditer(normalized_query):
        year = int(match.group(0))
        if claim(match):
            add(match, date(year, 1, 1), date(year, 12, 31), "absolute", "year")

    # Relative expressions, resolved against the dataset's own "today".
    relative: list[tuple[str, date, date, str]] = []
    q = normalized_query
    if re.search(r"\byesterday\b", q):
        d = reference_date - timedelta(days=1)
        relative.append(("yesterday", d, d, "day"))
    if re.search(r"\btoday\b", q):
        relative.append(("today", reference_date, reference_date, "day"))
    if re.search(r"\btomorrow\b", q):
        d = reference_date + timedelta(days=1)
        relative.append(("tomorrow", d, d, "day"))
    if re.search(r"\blast week\b|\bpast week\b", q):
        relative.append(("last week", reference_date - timedelta(days=7), reference_date, "span"))
    if re.search(r"\bnext week\b", q):
        relative.append(("next week", reference_date + timedelta(days=1),
                         reference_date + timedelta(days=7), "span"))
    if re.search(r"\blast month\b|\bpast month\b", q):
        relative.append(("last month", reference_date - timedelta(days=30), reference_date, "span"))
    if re.search(r"\bnext month\b", q):
        relative.append(("next month", reference_date + timedelta(days=1),
                         reference_date + timedelta(days=30), "span"))
    if re.search(r"\blast year\b", q):
        relative.append(("last year", reference_date - timedelta(days=365), reference_date, "span"))

    for match in _LOOKBACK_RE.finditer(q):
        days = int(match.group(1)) * _UNIT_DAYS[match.group(2)]
        relative.append((match.group(0), reference_date - timedelta(days=days), reference_date, "span"))

    for match in _HORIZON_RE.finditer(q):
        count = match.group(1) or match.group(3)
        unit = match.group(2) or match.group(4)
        if not count or not unit:
            continue
        days = int(count) * _UNIT_DAYS[unit]
        relative.append((match.group(0), reference_date + timedelta(days=1),
                         reference_date + timedelta(days=days), "span"))

    for raw, start, end, precision in relative:
        found.append(DateRef(raw, start, end, "relative", precision))

    return found


def extract_horizon_days(normalized_query: str) -> int | None:
    """How far ahead the question asks the model to look, in days."""
    horizons = []
    if re.search(r"\bnext week\b", normalized_query):
        horizons.append(7)
    if re.search(r"\btomorrow\b", normalized_query):
        horizons.append(1)
    if re.search(r"\bnext month\b", normalized_query):
        horizons.append(30)
    for match in _HORIZON_RE.finditer(normalized_query):
        count = match.group(1) or match.group(3)
        unit = match.group(2) or match.group(4)
        if count and unit:
            horizons.append(int(count) * _UNIT_DAYS[unit])
    return max(horizons) if horizons else None


# ── Metrics, comparisons, claims ─────────────────────────────────────────────

_METRIC_PATTERNS = {
    "sales": r"\bsales?\b|\brevenue\b|\bturnover\b|\btakings?\b",
    "customers": r"\bcustomers?\b|\bfootfall\b|\bvisitors?\b|\btransactions?\b",
    "promo_uplift": r"\bpromo\w*\b|\buplift\b|\bdiscount\w*\b",
    "forecast": r"\bforecast\w*\b|\bpredict\w*\b|\bexpected\b|\bprojection\b",
    "trend": r"\btrend\b|\bgrowth\b|\bdecline\b|\btrajectory\b|\bmomentum\b",
    "risk": r"\brisk\w*\b|\battention\b|\bconcern\w*\b|\bunderperform\w*\b",
    "drivers": r"\bdriver\w*\b|\bshap\b|\bwhy\b|\breason\w*\b|\bexplain\w*\b|\bcaus\w*\b",
    "anomaly": r"\banomal\w*\b|\bunusual\b|\boutlier\w*\b|\bspike\w*\b",
}

_COMPARISON_RE = re.compile(
    r"\bcompare[d]?\b|\bversus\b|\bvs\.?\b|\bagainst\b|\bbetween\b|\bdifference between\b"
    r"|\bbetter than\b|\bworse than\b|\brelative to\b",
    re.IGNORECASE,
)

# Anaphora: the question refers to something said earlier rather than naming it.
# Deliberately narrow. Bare "it"/"they"/"them" appear in ordinary questions
# ("instead of summarizing it"), and treating those as references to an earlier
# turn made the agent demand context it didn't need.
_CONTEXT_REFERENCE_RE = re.compile(
    r"\bwhich one\b|\bwhich of (?:them|those|these|the two|the ones)\b"
    r"|\bwhich should i\b|\bwhich (?:needs?|requires?|deserves?)\b"
    r"|\bwhich is (?:worse|better|riskier|weaker|stronger|more|less)\b"
    r"|\bthat store\b|\bthis store\b|\bthose stores?\b|\bthese stores?\b"
    r"|\bthe (?:two|three|both)\b|\bboth of them\b|\b(?:of|between|among) them\b"
    r"|\b(?:compare|rank|prioriti[sz]e|focus on)\s+(?:them|those|these)\b"
    r"|\bthe (?:first|second|last) one\b|\bthe same store\b"
    r"|\bmentioned (?:above|earlier|before)\b|\bwe (?:just )?(?:discussed|talked about)\b",
    re.IGNORECASE,
)

_RAW_DATA_RE = re.compile(
    r"\braw\s+(?:sales\s+)?(?:data|records?|rows?|table)\b"
    r"|\b(?:complete|full|entire)\s+(?:raw\s+)?(?:sales\s+)?(?:data|dataset|history|records?|rows?)\b"
    r"|\bevery\s+(?:store'?s?|single)\s+(?:complete|full|raw|entire|individual)\b"
    r"|\brow[- ]level\b|\brecord[- ]level\b|\btransaction[- ]level\b|\bline items?\b"
    r"|\bunaggregated\b|\binstead of summar\w*\b|\bwithout summar\w*\b|\bdo not summar\w*\b"
    r"|\ball (?:the )?(?:rows|records)\b",
    re.IGNORECASE,
)

_UP_WORDS = r"increase[d]?|increasing|rose|rise|risen|jump(?:ed)?|grew|grow(?:n|th)?|surge[d]?|climb(?:ed)?|up|gain(?:ed)?|spike[d]?"
_DOWN_WORDS = r"decrease[d]?|decreasing|drop(?:ped)?|declin(?:e|ed|ing)|fell|fall(?:en)?|down|dip(?:ped)?|lost|loss|slump(?:ed)?|crash(?:ed)?"

# Two orderings, because both are natural: "increased 83%" and "an 83% increase".
_CLAIM_PCT_AFTER_RE = re.compile(
    rf"\b({_UP_WORDS}|{_DOWN_WORDS})\b(?:\s+(?:by|of|about|around|roughly))?\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_CLAIM_PCT_BEFORE_RE = re.compile(
    rf"\b(\d+(?:\.\d+)?)\s*%\s*(?:\w+\s+){{0,2}}?\b({_UP_WORDS}|{_DOWN_WORDS})\b",
    re.IGNORECASE,
)
# "exactly ₹872431 in sales" — an asserted level, not a change.
_CLAIM_LEVEL_RE = re.compile(
    r"\b(?:exactly|precisely|was|were|had|did)\s*(?:[₹€$£]|rs\.?|inr|eur|usd)?\s*(\d{3,}(?:\.\d+)?)\b",
    re.IGNORECASE,
)

_UP_RE = re.compile(rf"^(?:{_UP_WORDS})$", re.IGNORECASE)


@dataclass(frozen=True)
class Claim:
    """A factual assertion the user made, which the agent has to check rather
    than accept. "Why did Store 125's sales increase 83% yesterday?" smuggles
    in a premise; answering the "why" without testing the "did" is how the
    agent ended up explaining an increase that never happened."""
    raw: str
    kind: str            # "change" | "level"
    direction: str       # "up" | "down" | ""
    value: float
    unit: str            # "percent" | "amount"


def extract_claims(normalized_query: str) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[tuple] = set()

    def add(claim: Claim) -> None:
        key = (claim.kind, claim.direction, claim.value, claim.unit)
        if key not in seen:
            seen.add(key)
            claims.append(claim)

    for match in _CLAIM_PCT_AFTER_RE.finditer(normalized_query):
        word, value = match.group(1), float(match.group(2))
        add(Claim(match.group(0), "change", "up" if _UP_RE.match(word) else "down", value, "percent"))

    for match in _CLAIM_PCT_BEFORE_RE.finditer(normalized_query):
        value, word = float(match.group(1)), match.group(2)
        add(Claim(match.group(0), "change", "up" if _UP_RE.match(word) else "down", value, "percent"))

    for match in _CLAIM_LEVEL_RE.finditer(normalized_query):
        add(Claim(match.group(0), "level", "", float(match.group(1)), "amount"))

    return claims


def extract_metrics(normalized_query: str) -> list[str]:
    return [name for name, pattern in _METRIC_PATTERNS.items()
            if re.search(pattern, normalized_query, re.IGNORECASE)]


# ── Fleet-level requests ─────────────────────────────────────────────────────

_RANKING_RE = re.compile(r"\b(top|bottom|best|worst|highest|lowest|rank(?:ed|ing)?)\b", re.IGNORECASE)
_WORST_RE = re.compile(r"\b(bottom|worst|lowest|underperform\w*|weakest)\b", re.IGNORECASE)
_PROMO_RE = re.compile(r"\b(promo\w*|uplift)\b", re.IGNORECASE)
_COUNT_RE = re.compile(
    r"\b(?:top|bottom|best|worst|first|last)\s+(\d{1,3})\b|\b(\d{1,3})\s+(?:stores?|performers?)\b",
    re.IGNORECASE,
)
_PLURAL_RE = re.compile(r"\b(stores|performers|shops|outlets|ones)\b", re.IGNORECASE)

_DEFAULT_RANKING_SIZE = 10
_MAX_RANKING_SIZE = 25


def parse_fleet_request(query: str) -> dict:
    """Which fleet-wide report a store-less question is asking for.

    How many rows to return follows the phrasing, so the answer matches the
    question: an explicit count wins ("top 15" -> 15); otherwise plural asks
    for a list ("best stores" -> 10) and singular asks for one ("best store
    among all" -> just that store).
    """
    wants_ranking = bool(_RANKING_RE.search(query))
    match = _COUNT_RE.search(query)
    requested = next((int(g) for g in (match.groups() if match else []) if g), None)

    if _PROMO_RE.search(query) and wants_ranking:
        kind = "promo_ranking"
    elif wants_ranking or requested is not None:
        kind = "sales_ranking"
    else:
        kind = "summary"

    if requested is not None:
        size = min(requested, _MAX_RANKING_SIZE)
    elif _PLURAL_RE.search(query):
        size = _DEFAULT_RANKING_SIZE
    else:
        size = 1

    return {"kind": kind, "n": size, "ascending": bool(_WORST_RE.search(query))}


# ── Scope ────────────────────────────────────────────────────────────────────

_RETAIL_VOCAB = (
    "store", "sales", "sale", "revenue", "forecast", "predict", "promo",
    "promotion", "uplift", "customer", "trend", "fleet", "performance",
    "performing", "perform", "anomaly", "underperform", "shap", "rmspe",
    "footfall", "assortment", "storetype", "holiday",
)


def is_in_scope(query: str, store_candidates: list[int]) -> bool:
    """A named store always counts; otherwise the question has to be
    recognisably about this retail dataset."""
    if store_candidates:
        return True
    q = query.lower()
    if any(term in q for term in _RETAIL_VOCAB):
        return True
    # "top 15" / "bottom 5" on its own is ranking phrasing, which in a store
    # analytics tool means stores.
    return bool(_COUNT_RE.search(query) and _RANKING_RE.search(query))


# ── Per-clause intent ────────────────────────────────────────────────────────

# A what-if is any hypothetical, not just the four phrasings the old keyword
# list happened to contain. "if a promotion were active" is the exact query
# that used to fall through to the forecast node and answer "no forecast".
_HYPOTHETICAL_RE = re.compile(
    r"\bwhat if\b|\bwhat would happen\b|\bwhat happens\b|\bwhat-if\b|\bsimulat\w*\b"
    r"|\bscenario\b|\bhypothetical\w*\b|\bsuppose\b|\bassuming\b|\bimagine\b"
    # The subject after "if" is any short noun phrase, not a fixed pronoun list:
    # "if a promotion were active" was matched, "if Store 125 runs a promotion"
    # was not, and both are the same question.
    r"|\bif\s+(?:\w+\s+){0,3}?(?:runs?|ran|add(?:s|ed)?|remov(?:e|es|ed)|turn(?:s|ed)?"
    r"|switch(?:es|ed)?|activat\w*|enabl\w*|disabl\w*|start(?:s|ed)?|were|was|is|are|had|have|has)\b"
    r"|\bwere (?:active|running|on|enabled|in place)\b|\bwould (?:sales|revenue|the forecast|it)\b",
    re.IGNORECASE,
)

# "Compare it with no promotion" is the second half of a what-if, not a request
# to rank stores. Both halves have to be present: a comparison *and* an explicit
# promo-on/promo-off contrast, so "how is Store 125 doing with promotions?"
# stays an ordinary performance question.
_SCENARIO_CONTRAST_RE = re.compile(
    r"\b(?:with|without)\s+(?:a\s+|the\s+|any\s+|no\s+)?promo\w*\b"
    r"|\bno\s+promo\w*\b|\bpromo\w*\s+(?:on|off)\b",
    re.IGNORECASE,
)
_PROMO_SUBJECT_RE = re.compile(r"\bpromo\w*\b|\bdiscount\w*\b|\bdeal\b|\boffer\b", re.IGNORECASE)
_FORECAST_RE = re.compile(
    r"\bforecast\w*\b|\bpredict\w*\b|\bexpect\w*\b|\bnext week\b|\bnext \d+ days?\b"
    r"|\bprojected?\b|\boutlook\b|\bgoing to (?:sell|make|do)\b|\bwill .{0,20}\bbe\b",
    re.IGNORECASE,
)
_RECOMMEND_RE = re.compile(
    r"\bfocus\b|\brecommend\w*\b|\bpriorit\w*\b|\bshould i\b|\bwhich (?:store|one)\b"
    r"|\bneeds? attention\b|\bwhat should\b|\bwhere should\b|\badvice\b|\baction\b",
    re.IGNORECASE,
)
# Risk language is checked before forecast language: "which 5 stores are most
# at risk of underperforming next week" mentions the horizon, but it is asking
# for the risk ranking the decision engine already computes, not a forecast.
_RISK_RE = re.compile(
    r"\bat risk\b|\brisk of\b|\brisk(?:iest)?\b|\bunderperform\w*\b|\bin trouble\b"
    r"|\bstruggl\w*\b|\bdeclin\w*\b|\bworr\w*\b|\bneed\w*\s+attention\b|\bmost vulnerable\b",
    re.IGNORECASE,
)


def classify_clause(clause: str, store_candidates: list[int], stores_named_here: int | None = None) -> str:
    """Intent for one clause. Deterministic and code-owned: an LLM asked to
    pick from a fixed label set always picks something, so scope and routing
    can't be delegated to it."""
    if _RAW_DATA_RE.search(clause):
        return "raw_data"
    if not is_in_scope(clause, store_candidates):
        return "out_of_scope"
    if _HYPOTHETICAL_RE.search(clause) and (_PROMO_SUBJECT_RE.search(clause) or _FORECAST_RE.search(clause)):
        return "whatif"
    if _SCENARIO_CONTRAST_RE.search(clause) and _COMPARISON_RE.search(clause):
        return "whatif"
    # Checked before forecast keywords: "which of these should I focus on next
    # week" mentions the horizon but is asking for a ranking, not a forecast.
    # Counted from this clause alone: "and what is the promo uplift?" inherits
    # the two stores named earlier in the question, but it is not itself a
    # request to rank them.
    named_here = len(store_candidates) if stores_named_here is None else stores_named_here
    if _RISK_RE.search(clause) and _RANKING_RE.search(clause) or _RISK_RE.search(clause) and not store_candidates:
        return "recommend"
    if _RECOMMEND_RE.search(clause) or _COMPARISON_RE.search(clause) or named_here > 1:
        return "recommend"
    if _FORECAST_RE.search(clause):
        return "forecast"
    return "performance"


# ── The assembled understanding ──────────────────────────────────────────────

@dataclass
class IntentSpec:
    """One thing the user asked for. A query produces a list of these, so a
    six-part question stays six parts all the way through the pipeline."""
    type: str
    stores: list[int] = field(default_factory=list)
    scope: str = "fleet"          # "store" | "fleet"
    clause: str = ""
    metrics: list[str] = field(default_factory=list)
    # Dates named in *this* clause. Held per step because one stacked question
    # can ask for a valid 7-day forecast and a 2027 forecast at the same time —
    # only the second is unanswerable, and merging them loses the first.
    dates: list[DateRef] = field(default_factory=list)

    def date_signature(self) -> tuple:
        return tuple(sorted((d.start, d.end) for d in self.dates))


@dataclass
class Entities:
    store_candidates: list[int] = field(default_factory=list)
    dates: list[DateRef] = field(default_factory=list)
    horizon_days: int | None = None
    metrics: list[str] = field(default_factory=list)
    comparison: bool = False
    claims: list[Claim] = field(default_factory=list)
    wants_raw_records: bool = False
    fleet_request: dict = field(default_factory=dict)
    references_prior_context: bool = False
    exceeds_store_cap: bool = False


@dataclass
class Understanding:
    raw_query: str
    normalized: str
    clauses: list[str]
    intents: list[IntentSpec]
    entities: Entities
    resolved_from_context: bool = False

    @property
    def intent_types(self) -> list[str]:
        return [i.type for i in self.intents]

    @property
    def primary_intent(self) -> str:
        return self.intents[0].type if self.intents else "out_of_scope"

    @property
    def is_multi_intent(self) -> bool:
        return len(self.intents) > 1


def is_fleet_risk_request(clause: str) -> bool:
    """A risk question with no store named — "which stores are most at risk?"."""
    return bool(_RISK_RE.search(clause))


def understand(query: str, reference_date: date, context: dict | None = None,
               store_id_range: tuple[int, int] = (1, 1115)) -> Understanding:
    """Raw text -> structure, with earlier turns filled in where the question
    leans on them.

    `context` carries what the session already discussed (previous_stores,
    previous_intents, previous_metrics, previous_comparison,
    previous_date_range). "Which one should I prioritise?" names no store, so
    without it the question is unanswerable — the agent used to reply with a
    generic refusal even though it had just been told which stores mattered.
    """
    context = context or {}
    normalized = normalize(query)
    clauses = split_clauses(normalized)

    store_candidates = extract_store_candidates(normalized)
    if not store_candidates:
        # Only consulted when no store was named explicitly, so the explicit
        # form always wins and this can never override it.
        store_candidates = extract_bare_store_candidates(normalized, *store_id_range)
    references_context = bool(
        not store_candidates and _CONTEXT_REFERENCE_RE.search(normalized)
    )

    resolved_from_context = False
    if not store_candidates and references_context and context.get("previous_stores"):
        store_candidates = list(context["previous_stores"])
        resolved_from_context = True

    entities = Entities(
        store_candidates=store_candidates,
        dates=extract_dates(normalized, reference_date),
        horizon_days=extract_horizon_days(normalized),
        metrics=extract_metrics(normalized),
        comparison=bool(_COMPARISON_RE.search(normalized)) or len(store_candidates) > 1,
        claims=extract_claims(normalized),
        wants_raw_records=bool(_RAW_DATA_RE.search(normalized)),
        fleet_request=parse_fleet_request(normalized),
        references_prior_context=references_context,
        exceeds_store_cap=mentions_more_stores_than(normalized),
    )

    if not entities.metrics and resolved_from_context and context.get("previous_metrics"):
        entities.metrics = list(context["previous_metrics"])

    intents: list[IntentSpec] = []
    for clause in clauses:
        named_here = extract_store_candidates(clause) or extract_bare_store_candidates(
            clause, *store_id_range)
        clause_stores = named_here or store_candidates
        intent_type = classify_clause(clause, clause_stores, stores_named_here=len(named_here))
        # A clause can only be out of scope if the whole query is: "and why?"
        # on its own carries no retail vocabulary but inherits the subject of
        # the question it was attached to.
        if intent_type == "out_of_scope" and len(clauses) > 1 and is_in_scope(normalized, store_candidates):
            intent_type = classify_clause(f"{clause} sales", clause_stores,
                                          stores_named_here=len(named_here))
        intents.append(IntentSpec(
            type=intent_type,
            stores=clause_stores[:MAX_STORES_PER_QUERY],
            scope="store" if clause_stores else "fleet",
            clause=clause,
            metrics=extract_metrics(clause),
            dates=extract_dates(clause, reference_date),
        ))

    # A stacked question that resolves to the same intent twice is still one
    # piece of work — "which should I focus on, and which is riskiest?".
    deduped: list[IntentSpec] = []
    for spec in intents:
        twin = next((d for d in deduped
                     if d.type == spec.type and d.stores == spec.stores
                     and d.date_signature() == spec.date_signature()), None)
        if twin is not None:
            # Same work, different wording — keep the metrics from both so the
            # answer still covers everything that was asked.
            twin.metrics = list(dict.fromkeys(twin.metrics + spec.metrics))
            continue
        deduped.append(spec)

    # Scope is decided over the whole query: one off-topic clause inside an
    # otherwise legitimate question shouldn't refuse the whole thing, and an
    # entirely off-topic question must not keep a stray in-scope-looking part.
    if not is_in_scope(normalized, store_candidates) and not entities.wants_raw_records:
        deduped = [IntentSpec(type="out_of_scope", stores=[], scope="fleet", clause=normalized)]
    else:
        in_scope = [s for s in deduped if s.type != "out_of_scope"]
        deduped = in_scope or [IntentSpec(type="performance", stores=store_candidates[:MAX_STORES_PER_QUERY],
                                          scope="store" if store_candidates else "fleet",
                                          clause=normalized, metrics=entities.metrics,
                                          dates=list(entities.dates))]

    if not store_candidates and context.get("previous_stores") and any(
        s.type in ("recommend", "whatif") and not s.stores for s in deduped
    ) and references_context:
        for spec in deduped:
            if not spec.stores:
                spec.stores = list(context["previous_stores"])[:MAX_STORES_PER_QUERY]
                spec.scope = "store"
        resolved_from_context = True

    return Understanding(
        raw_query=query,
        normalized=normalized,
        clauses=clauses,
        intents=deduped,
        entities=entities,
        resolved_from_context=resolved_from_context,
    )

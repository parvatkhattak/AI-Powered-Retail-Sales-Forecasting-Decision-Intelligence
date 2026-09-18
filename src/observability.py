"""
src/observability.py
Owner: Parvat — Team Lead / Integration

Responsibilities:
- Structured JSON logging to logs/app.log (rotating)
- FinOps: token counting and LLM cost estimation per call
- @timed decorator for latency instrumentation
- Health-check helper for all subsystems
- In-process metric accumulator (stored in memory + JSONL file)

Design principles:
- ZERO side-effects on import: nothing starts until init_observability() is called
- All public helpers fail silently — observability must NEVER crash the app
- Completely additive: existing code works identically if this module is never imported
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

# ── Resolve project root ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = _ROOT / "logs"

# ── Module-level state ────────────────────────────────────────────────────────
_obs_logger: logging.Logger = logging.getLogger("observability")
_finops_path: Optional[Path] = None
_metrics_path: Optional[Path] = None
_session_id: str = str(uuid.uuid4())[:8]

# In-memory metric store (reset on app restart, that's fine)
_metrics: dict[str, list[dict]] = {
    "llm_calls": [],
    "latency_events": [],
    "errors": [],
    "health_checks": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────

def init_observability(log_level: str = "INFO") -> None:
    """
    Call once at app startup (app.py) to activate structured logging and
    set up the log directory. Safe to call multiple times — idempotent.
    """
    global _finops_path, _metrics_path

    try:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _finops_path  = _LOGS_DIR / "finops.jsonl"
        _metrics_path = _LOGS_DIR / "metrics.jsonl"

        # ── Root app logger: JSON to rotating file + human text to console ──
        root = logging.getLogger()
        if root.handlers:          # already configured (e.g. Streamlit did it)
            pass
        else:
            root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Rotating file handler — structured JSON lines
        file_handler = logging.handlers.RotatingFileHandler(
            _LOGS_DIR / "app.log",
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JsonFormatter())
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Avoid adding duplicate handlers on Streamlit re-runs
        obs = logging.getLogger("observability")
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in obs.handlers):
            obs.addHandler(file_handler)
            obs.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            obs.propagate = False

        _obs_logger.info("observability_init", extra={"session_id": _session_id})

    except Exception:  # pragma: no cover
        pass  # Observability must never crash the app


# ─────────────────────────────────────────────────────────────────────────────
# JSON LOG FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line for structured parsing."""

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "session": _session_id,
            "msg":     record.getMessage(),
        }
        # Merge any extra fields added by the caller
        for key, val in record.__dict__.items():
            if key not in {
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "taskName",
                "thread", "threadName",
            }:
                doc[key] = val
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# @timed DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

def timed(event_name: str, extra: Optional[dict] = None) -> Callable:
    """
    Decorator that measures wall-clock latency of a function and records it.

    Usage:
        @timed("forecast_predict")
        def get_forecast(store_id):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            status = "ok"
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:
                status = "error"
                record_error(event_name, str(exc))
                raise
            finally:
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                _record_latency(event_name, latency_ms, status, extra or {})
        return wrapper
    return decorator


def _record_latency(event: str, latency_ms: float, status: str, extra: dict) -> None:
    """Append one latency observation to in-memory store and JSONL file."""
    try:
        entry = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "session":    _session_id,
            "event":      event,
            "latency_ms": latency_ms,
            "status":     status,
            **extra,
        }
        _metrics["latency_events"].append(entry)
        # Keep last 500 in memory
        if len(_metrics["latency_events"]) > 500:
            _metrics["latency_events"] = _metrics["latency_events"][-500:]

        if _metrics_path:
            with open(_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

        _obs_logger.debug(
            "latency", extra={"event": event, "latency_ms": latency_ms, "status": status}
        )
    except Exception:  # pragma: no cover
        pass


# ─────────────────────────────────────────────────────────────────────────────
# FINOPS — Token & Cost Tracking
# ─────────────────────────────────────────────────────────────────────────────

# OpenRouter pricing per 1M tokens (prompt / completion).
# Free-tier models are $0, but we track volume anyway.
_COST_TABLE: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.3-70b-instruct:free": (0.0,   0.0),
    "meta-llama/llama-3.1-8b-instruct:free":  (0.0,   0.0),
    "google/gemma-2-9b-it:free":               (0.0,   0.0),
    "openai/gpt-4o-mini":                      (0.15,  0.60),
    "openai/gpt-4o":                           (5.00, 15.00),
    "anthropic/claude-3-haiku":                (0.25,  1.25),
}


def record_llm_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    status: str = "ok",
    query_preview: str = "",
) -> None:
    """
    Call this after every LLM invocation. Computes cost and stores to
    in-memory store + finops.jsonl.

    All args have safe defaults so the call never raises.
    """
    try:
        prompt_rate, completion_rate = _COST_TABLE.get(model, (0.0, 0.0))
        cost_usd = (
            prompt_tokens     * prompt_rate     / 1_000_000
            + completion_tokens * completion_rate / 1_000_000
        )
        entry = {
            "ts":                datetime.now(timezone.utc).isoformat(),
            "session":           _session_id,
            "model":             model,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "cost_usd":          round(cost_usd, 8),
            "latency_ms":        round(latency_ms, 1),
            "status":            status,
            "query_preview":     query_preview[:80],
        }
        _metrics["llm_calls"].append(entry)
        if len(_metrics["llm_calls"]) > 200:
            _metrics["llm_calls"] = _metrics["llm_calls"][-200:]

        if _finops_path:
            with open(_finops_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

        _obs_logger.info(
            "llm_call",
            extra={
                "model":        model,
                "total_tokens": entry["total_tokens"],
                "cost_usd":     entry["cost_usd"],
                "latency_ms":   round(latency_ms, 1),
                "status":       status,
            },
        )
    except Exception:  # pragma: no cover
        pass


def record_error(event: str, message: str) -> None:
    """Record an application error for the dashboard."""
    try:
        entry = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "session": _session_id,
            "event":   event,
            "message": message[:200],
        }
        _metrics["errors"].append(entry)
        if len(_metrics["errors"]) > 100:
            _metrics["errors"] = _metrics["errors"][-100:]
        _obs_logger.error("app_error", extra={"event": event, "message": message[:200]})
    except Exception:  # pragma: no cover
        pass


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def run_health_check() -> dict[str, dict]:
    """
    Returns a dict of subsystem → {status: ok|warn|error, detail: str}.
    Never raises — each check is individually try/except'd.
    """
    results: dict[str, dict] = {}

    # 1. Database
    try:
        import sqlite3
        sys.path.insert(0, str(_ROOT))
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        conn.close()
        results["database"] = {"status": "ok", "detail": f"{count:,} sales rows"}
    except Exception as exc:
        results["database"] = {"status": "error", "detail": str(exc)[:120]}

    # 2. ML models
    try:
        from config import LGBM_PATH, MODEL_PATH
        missing = [p.name for p in [MODEL_PATH, LGBM_PATH] if not p.exists()]
        if missing:
            results["ml_models"] = {"status": "warn", "detail": f"Missing: {missing}"}
        else:
            results["ml_models"] = {"status": "ok", "detail": "lgbm + xgboost loaded"}
    except Exception as exc:
        results["ml_models"] = {"status": "error", "detail": str(exc)[:120]}

    # 3. LLM API key
    try:
        from config import LLM_API_KEY
        if LLM_API_KEY and len(LLM_API_KEY) > 10:
            results["llm_api"] = {"status": "ok", "detail": "API key set"}
        else:
            results["llm_api"] = {"status": "warn", "detail": "OPENROUTER_API_KEY not set"}
    except Exception as exc:
        results["llm_api"] = {"status": "error", "detail": str(exc)[:120]}

    # 4. Log directory
    try:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        results["log_dir"] = {"status": "ok", "detail": str(_LOGS_DIR)}
    except Exception as exc:
        results["log_dir"] = {"status": "error", "detail": str(exc)[:120]}

    # 5. Python / platform
    results["runtime"] = {
        "status": "ok",
        "detail": f"Python {sys.version.split()[0]} | PID {os.getpid()}",
    }

    ts = datetime.now(timezone.utc).isoformat()
    _metrics["health_checks"].append({"ts": ts, "results": results})
    if len(_metrics["health_checks"]) > 50:
        _metrics["health_checks"] = _metrics["health_checks"][-50:]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# METRIC ACCESSORS (used by the dashboard page)
# ─────────────────────────────────────────────────────────────────────────────

def get_session_summary() -> dict:
    """Aggregate stats for the current app session."""
    try:
        llm_calls = _metrics["llm_calls"]
        latency   = _metrics["latency_events"]
        errors    = _metrics["errors"]

        total_tokens   = sum(c.get("total_tokens",     0) for c in llm_calls)
        total_cost     = sum(c.get("cost_usd",         0) for c in llm_calls)
        total_llm_calls= len(llm_calls)
        avg_latency_ms = (
            sum(e.get("latency_ms", 0) for e in latency) / len(latency)
            if latency else 0
        )
        error_count = len(errors)

        return {
            "session_id":       _session_id,
            "total_llm_calls":  total_llm_calls,
            "total_tokens":     total_tokens,
            "total_cost_usd":   round(total_cost, 6),
            "avg_latency_ms":   round(avg_latency_ms, 1),
            "error_count":      error_count,
            "llm_calls":        llm_calls,
            "latency_events":   latency,
            "errors":           errors,
        }
    except Exception:  # pragma: no cover
        return {}


def get_log_tail(n: int = 50) -> list[dict]:
    """Read last N lines from app.log and parse them as JSON dicts."""
    try:
        log_file = _LOGS_DIR / "app.log"
        if not log_file.exists():
            return []
        lines = log_file.read_text(encoding="utf-8").splitlines()
        tail = lines[-n:] if len(lines) >= n else lines
        result = []
        for line in reversed(tail):
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                result.append({"msg": line})
        return result
    except Exception:  # pragma: no cover
        return []

"""
Structured audit logger for the evaluation pipeline.
- Step-by-step console logging for each evaluation stage.
- JSONL audit file for full results.
- Pipeline summary for batch runs.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
AUDIT_LOG_PATH = LOG_DIR / "evaluation_audit.jsonl"


def _get_console_logger() -> logging.Logger:
    """Returns a human-readable console logger."""
    logger = logging.getLogger("syd_life_eval")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


_console = _get_console_logger()


# ─── Step-by-step evaluation logging ───────────────────────────────────────

def log_evaluation_start(transcript_id: str, title: str) -> None:
    """Log the start of an evaluation for a transcript."""
    _console.info(f"Starting evaluation for [{transcript_id}] {title!r}")
    _console.info("  " + "─" * 50)


def log_step_begin(step_num: int, step_name: str, description: str) -> None:
    """Log the beginning of a pipeline step."""
    _console.info(f"  Step {step_num}: {step_name} — {description}")


def log_step_end(step_num: int, step_name: str, outcome: str, detail: str | None = None) -> None:
    """Log the outcome of a pipeline step."""
    if detail:
        _console.info(f"         → {outcome} | {detail}")
    else:
        _console.info(f"         → {outcome}")


def log_step_safety_result(safe: bool, reasoning_preview: str | None = None) -> None:
    """Log medical safety step result."""
    outcome = "safe=True" if safe else "safe=False (VIOLATION)"
    if reasoning_preview:
        preview = (reasoning_preview[:120] + "…") if len(reasoning_preview) > 120 else reasoning_preview
        log_step_end(1, "Medical Safety", outcome, preview)
    else:
        log_step_end(1, "Medical Safety", outcome)


def log_step_empathy_result(level: str, passed: bool) -> None:
    """Log empathy step result (level E0–E3 + label)."""
    from evaluator.core import EMPATHY_LEVEL_LABELS
    label = EMPATHY_LEVEL_LABELS.get(level, level)
    outcome = f"level={level} — {label}, passed={passed}"
    log_step_end(2, "Empathy", outcome)


def log_step_groundedness_result(level: str, passed: bool, refs: list[str], hallucinated: list[str]) -> None:
    """Log groundedness step result (level G0–G4 + label), including referenced chunks and any hallucinated claims."""
    from evaluator.core import GROUNDEDNESS_LEVEL_LABELS
    label = GROUNDEDNESS_LEVEL_LABELS.get(level, level)
    outcome = f"level={level} — {label}, passed={passed}"
    detail_parts = []
    if refs:
        detail_parts.append(f"referenced={refs[:5]}{'…' if len(refs) > 5 else ''}")
    if hallucinated:
        detail_parts.append(f"hallucinated_claims={len(hallucinated)}")
    log_step_end(3, "Groundedness", outcome, "; ".join(detail_parts) if detail_parts else None)
    if hallucinated:
        for claim in hallucinated:
            preview = (claim[:100] + "…") if len(claim) > 100 else claim
            _console.warning(f"         ⚠ hallucinated: {preview}")


def log_step_verdict(verdict: Any, flags: list[Any]) -> None:
    """Log the computed verdict and flags."""
    v_str = verdict.value if hasattr(verdict, "value") else str(verdict)
    f_list = [f.value if hasattr(f, "value") else str(f) for f in flags]
    outcome = f"verdict={v_str}"
    detail = f"flags={f_list}" if f_list else None
    log_step_end(4, "Verdict", outcome, detail)


def log_evaluation_result(result: Any) -> None:
    """
    Log the final evaluation result (summary + JSONL audit).
    Call this after all steps are done.
    """
    record = result.model_dump()
    record["logged_at"] = datetime.now(timezone.utc).isoformat()

    # Append to JSONL audit file
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Console: final result section
    verdict_val = result.verdict.value if hasattr(result.verdict, "value") else result.verdict
    symbol = {"PASS": "✅", "FAIL": "❌", "HARD_FAIL": "🚨"}.get(verdict_val, "?")
    _console.info("  " + "─" * 50)
    _console.info(f"  RESULT: {symbol} [{result.transcript_id}] {result.title} → {verdict_val}")
    from evaluator.core import GROUNDEDNESS_LEVEL_LABELS, EMPATHY_LEVEL_LABELS
    g_label = GROUNDEDNESS_LEVEL_LABELS.get(result.groundedness.level, result.groundedness.level)
    e_label = EMPATHY_LEVEL_LABELS.get(result.empathy.level, result.empathy.level)
    _console.info(
        f"         Empathy: {result.empathy.level} — {e_label} | "
        f"Groundedness: {result.groundedness.level} — {g_label} | "
        f"Medical Safety: {result.medical_safety.safe}"
    )
    if result.flags:
        _console.info(f"         Flags: {[f.value for f in result.flags]}")
    if result.medical_safety.violation_excerpt:
        excerpt = (result.medical_safety.violation_excerpt[:100] + "…") if len(result.medical_safety.violation_excerpt) > 100 else result.medical_safety.violation_excerpt
        _console.warning(f"         Safety excerpt: {excerpt!r}")
    if result.groundedness.hallucinated_claims:
        for claim in result.groundedness.hallucinated_claims[:3]:
            _console.warning(f"         ⚠ Hallucinated: {claim[:80]}…" if len(claim) > 80 else f"         ⚠ Hallucinated: {claim}")
        if len(result.groundedness.hallucinated_claims) > 3:
            _console.warning(f"         … and {len(result.groundedness.hallucinated_claims) - 3} more")
    _console.info("")


def log_pipeline_summary(results: list[Any]) -> None:
    """Log summary for a batch run and append PIPELINE_SUMMARY to audit file."""
    total = len(results)
    passed = sum(1 for r in results if (r.verdict.value if hasattr(r.verdict, "value") else r.verdict) == "PASS")
    failed = sum(1 for r in results if (r.verdict.value if hasattr(r.verdict, "value") else r.verdict) == "FAIL")
    hard_failed = sum(1 for r in results if (r.verdict.value if hasattr(r.verdict, "value") else r.verdict) == "HARD_FAIL")

    summary = {
        "type": "PIPELINE_SUMMARY",
        "pipeline_run_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "hard_failed": hard_failed,
        "pass_rate": round(passed / total, 3) if total else 0,
    }

    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    _console.info("=" * 60)
    _console.info(
        f"PIPELINE SUMMARY | Total: {total} | ✅ Pass: {passed} | "
        f"❌ Fail: {failed} | 🚨 Hard Fail: {hard_failed} | "
        f"Pass rate: {summary['pass_rate']:.1%}"
    )
    _console.info(f"Audit log: {AUDIT_LOG_PATH.resolve()}")
    _console.info("")


def log_error(context: str, error: Exception) -> None:
    """Log an error with context to console and audit file."""
    record = {
        "type": "ERROR",
        "context": context,
        "error": str(error),
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    _console.error(f"ERROR in {context}: {error}")

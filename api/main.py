"""
FastAPI application — Syd Life AI Evaluation Pipeline
Exposes endpoints to evaluate individual transcripts or run full batch evaluation.
"""

import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from data.loaders import load_transcripts
from evaluator import (
    EvaluationResult,
    JudgeLLM,
    JudgeTimeoutError,
    log_error,
    log_evaluation_result,
    log_pipeline_summary,
)
from evaluator.core import EMPATHY_LEVEL_LABELS, GROUNDEDNESS_LEVEL_LABELS

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Syd Life AI — LLM Evaluation Pipeline",
    description=(
        "Automated auditor that evaluates preventive health AI agent responses "
        "for Empathy, Groundedness, and Medical Safety."
    ),
    version="1.0.0",
)

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
CORS_ORIGINS_STR = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_STR.split(",") if o.strip()] if CORS_ORIGINS_STR != "*" else ["*"]
RATE_LIMIT = os.environ.get("RATE_LIMIT", "30/minute")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_transcript_store() -> dict:
    transcripts = load_transcripts()
    return {"transcripts": transcripts, "index": {t["transcript_id"]: t for t in transcripts}}

_transcript_store = _load_transcript_store()

def _get_transcripts() -> list[dict]:
    return _transcript_store["transcripts"]

def _get_transcript_index() -> dict[str, dict]:
    return _transcript_store["index"]

def _reload_transcripts() -> None:
    global _transcript_store
    _transcript_store = _load_transcript_store()

def _error_detail(e: Exception) -> str:
    if DEBUG:
        return str(e)
    return "Internal server error. Set DEBUG=1 for details."


# ── Request / Response models ─────────────────────────────────────────────────

class TranscriptTurn(BaseModel):
    role: str  # "user" or "agent"
    content: str


class RetrievedChunk(BaseModel):
    """Optional chunk for RAG-style groundedness (evaluate agent against this context only)."""
    chunk_id: str
    text: str
    source: str = ""
    retrieval_score: Optional[float] = None


class EvaluateRequest(BaseModel):
    turns: list[TranscriptTurn]
    retrieved_chunks: Optional[list[RetrievedChunk]] = None  # If provided, groundedness is judged against these chunks only


class BatchEvaluateResponse(BaseModel):
    attempted: int
    total: int
    passed: int
    failed: int
    hard_failed: int
    pass_rate: float
    results: list[EvaluationResult]
    errors: list[dict] = []  # [{ "transcript_id": str, "error": str }, ...]


def _wrap(text: str, width: int = 70) -> list[str]:
    """Simple line wrap for report paragraphs."""
    words = text.split()
    lines = []
    current: list[str] = []
    current_len = 0
    for w in words:
        if current_len + len(w) + (1 if current else 0) <= width:
            current.append(w)
            current_len += len(w) + (1 if len(current) > 1 else 0)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
    if current:
        lines.append(" ".join(current))
    return lines


def build_evaluation_report(result: EvaluationResult) -> str:
    """Format an EvaluationResult as a human-readable report for presentation."""
    verdict_val = result.verdict.value if hasattr(result.verdict, "value") else result.verdict
    verdict_icon = {"PASS": "✅", "FAIL": "❌", "HARD_FAIL": "🚨"}.get(verdict_val, "—")
    e_label = EMPATHY_LEVEL_LABELS.get(result.empathy.level, result.empathy.level)
    g_label = GROUNDEDNESS_LEVEL_LABELS.get(result.groundedness.level, result.groundedness.level)

    lines = [
        "═══════════════════════════════════════════════════════════════",
        "  EVALUATION REPORT — Syd Life AI",
        "═══════════════════════════════════════════════════════════════",
        "",
        f"  Transcript: {result.title}",
        f"  ID: {result.transcript_id}",
        f"  Verdict: {verdict_icon} {verdict_val}",
        f"  Evaluated: {result.evaluation_timestamp}",
        f"  Model: {result.model_used}",
        "",
        "───────────────────────────────────────────────────────────────",
        "  1. MEDICAL SAFETY",
        "───────────────────────────────────────────────────────────────",
        "",
        f"  Status: {'✓ Safe' if result.medical_safety.safe else '✗ Violation'}",
        "",
        "  Reasoning:",
        *[f"    {p}" for p in _wrap(result.medical_safety.reasoning, width=66)],
        "",
    ]
    if result.medical_safety.violation_excerpt:
        lines.extend([
            "  Violation excerpt:",
            f"    \"{result.medical_safety.violation_excerpt[:200]}{'…' if len(result.medical_safety.violation_excerpt) > 200 else ''}\"",
            "",
        ])

    lines.extend([
        "───────────────────────────────────────────────────────────────",
        "  2. EMPATHY",
        "───────────────────────────────────────────────────────────────",
        "",
        f"  Level: {result.empathy.level} — {e_label}",
        f"  Passed: {'Yes' if result.empathy.passed else 'No'}",
        "",
        "  Reasoning:",
        *[f"    {p}" for p in _wrap(result.empathy.reasoning, width=66)],
        "",
        "───────────────────────────────────────────────────────────────",
        "  3. GROUNDEDNESS",
        "───────────────────────────────────────────────────────────────",
        "",
        f"  Level: {result.groundedness.level} — {g_label}",
        f"  Passed: {'Yes' if result.groundedness.passed else 'No'}",
        "",
        "  Reasoning:",
        *[f"    {p}" for p in _wrap(result.groundedness.reasoning, width=66)],
        "",
    ])
    if result.groundedness.referenced_guidelines:
        lines.append("  Referenced guidelines: " + ", ".join(result.groundedness.referenced_guidelines))
        lines.append("")
    if result.groundedness.hallucinated_claims:
        lines.append("  Hallucinated / ungrounded claims:")
        for c in result.groundedness.hallucinated_claims:
            lines.append(f"    • {c[:120]}{'…' if len(c) > 120 else ''}")
        lines.append("")

    if result.flags:
        lines.extend([
            "───────────────────────────────────────────────────────────────",
            "  FLAGS",
            "───────────────────────────────────────────────────────────────",
            "",
            "  " + ", ".join(f.value if hasattr(f, "value") else str(f) for f in result.flags),
            "",
        ])

    lines.extend([
        "═══════════════════════════════════════════════════════════════",
        "  End of report",
        "═══════════════════════════════════════════════════════════════",
    ])
    return "\n".join(lines)


class EvaluationReportResponse(BaseModel):
    """Evaluation result plus a human-readable report for presentation."""
    transcript_id: str
    title: str
    empathy: dict
    groundedness: dict
    medical_safety: dict
    flags: list[str]
    verdict: str
    model_used: str
    evaluation_timestamp: str
    report: str


def _result_to_report_response(result: EvaluationResult) -> EvaluationReportResponse:
    """Build API response with structured fields plus human-readable report."""
    return EvaluationReportResponse(
        transcript_id=result.transcript_id,
        title=result.title,
        empathy=result.empathy.model_dump(),
        groundedness=result.groundedness.model_dump(),
        medical_safety=result.medical_safety.model_dump(),
        flags=[f.value if hasattr(f, "value") else str(f) for f in result.flags],
        verdict=result.verdict.value if hasattr(result.verdict, "value") else result.verdict,
        model_used=result.model_used,
        evaluation_timestamp=result.evaluation_timestamp,
        report=build_evaluation_report(result),
    )


def _require_openai_key() -> None:
    """Raise 503 if OpenAI API key is not configured (avoids leaking KeyError as 500)."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY in .env or environment.",
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "Syd Life AI Evaluation Pipeline"}


@app.get("/transcripts", summary="List all available mock transcripts")
def list_transcripts():
    return [
        {
            "transcript_id": t["transcript_id"],
            "title": t["title"],
            "expected_verdict": t.get("expected_verdict"),
            "expected_flags": t.get("expected_flags", []),
        }
        for t in _get_transcripts()
    ]


@app.post("/transcripts/reload", summary="Reload transcripts from data/transcripts.json")
def reload_transcripts():
    """Reload transcripts from disk without restarting the server."""
    try:
        _reload_transcripts()
        n = len(_get_transcripts())
        return {"status": "ok", "message": f"Reloaded {n} transcripts."}
    except Exception as e:
        log_error("reload_transcripts", e)
        raise HTTPException(status_code=500, detail=_error_detail(e))


@app.get(
    "/evaluate/{transcript_id}",
    response_class=PlainTextResponse,
    summary="Evaluate a transcript and return a human-readable report (default)",
)
@limiter.limit(RATE_LIMIT)
def evaluate_by_id(request: Request, transcript_id: str, model: Optional[str] = Query(default=None)):
    """Returns the formatted report (text/plain). Use GET .../evaluate/{id}/json for JSON."""
    _require_openai_key()
    transcript = _get_transcript_index().get(transcript_id)
    if not transcript:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript '{transcript_id}' not found. Use GET /transcripts to list available IDs.",
        )
    try:
        judge = JudgeLLM(model=model)
        result = judge.evaluate(transcript)
        log_evaluation_result(result)
        return build_evaluation_report(result)
    except JudgeTimeoutError as e:
        log_error(f"evaluate_by_id:{transcript_id}", e)
        raise HTTPException(status_code=504, detail=_error_detail(e))
    except Exception as e:
        log_error(f"evaluate_by_id:{transcript_id}", e)
        raise HTTPException(status_code=500, detail=_error_detail(e))


@app.get(
    "/evaluate/{transcript_id}/json",
    response_model=EvaluationReportResponse,
    summary="Evaluate a transcript and return JSON (structured result + report field)",
)
@limiter.limit(RATE_LIMIT)
def evaluate_by_id_json(request: Request, transcript_id: str, model: Optional[str] = Query(default=None)):
    transcript = _get_transcript_index().get(transcript_id)
    if not transcript:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript '{transcript_id}' not found. Use GET /transcripts to list available IDs.",
        )
    _require_openai_key()
    try:
        judge = JudgeLLM(model=model)
        result = judge.evaluate(transcript)
        log_evaluation_result(result)
        return _result_to_report_response(result)
    except JudgeTimeoutError as e:
        log_error(f"evaluate_by_id_json:{transcript_id}", e)
        raise HTTPException(status_code=504, detail=_error_detail(e))
    except Exception as e:
        log_error(f"evaluate_by_id_json:{transcript_id}", e)
        raise HTTPException(status_code=500, detail=_error_detail(e))


@app.post(
    "/evaluate",
    response_class=PlainTextResponse,
    summary="Evaluate a custom transcript and return a human-readable report (default)",
)
@limiter.limit(RATE_LIMIT)
def evaluate_custom(request: Request, body: EvaluateRequest, model: Optional[str] = Query(default=None)):
    """Returns the formatted report (text/plain). Use POST /evaluate/json for JSON."""
    _require_openai_key()
    transcript = {
        "transcript_id": "CUSTOM",
        "title": "Custom Submission",
        "turns": [t.model_dump() for t in body.turns],
    }
    if body.retrieved_chunks is not None:
        transcript["retrieved_chunks"] = [c.model_dump() for c in body.retrieved_chunks]
    try:
        judge = JudgeLLM(model=model)
        result = judge.evaluate(transcript)
        log_evaluation_result(result)
        return build_evaluation_report(result)
    except JudgeTimeoutError as e:
        log_error("evaluate_custom", e)
        raise HTTPException(status_code=504, detail=_error_detail(e))
    except Exception as e:
        log_error("evaluate_custom", e)
        raise HTTPException(status_code=500, detail=_error_detail(e))


@app.post(
    "/evaluate/json",
    response_model=EvaluationResult,
    summary="Evaluate a custom transcript and return JSON (for scripts)",
)
@limiter.limit(RATE_LIMIT)
def evaluate_custom_json(request: Request, body: EvaluateRequest, model: Optional[str] = Query(default=None)):
    _require_openai_key()
    transcript = {
        "transcript_id": "CUSTOM",
        "title": "Custom Submission",
        "turns": [t.model_dump() for t in body.turns],
    }
    if body.retrieved_chunks is not None:
        transcript["retrieved_chunks"] = [c.model_dump() for c in body.retrieved_chunks]
    try:
        judge = JudgeLLM(model=model)
        result = judge.evaluate(transcript)
        log_evaluation_result(result)
        return result
    except JudgeTimeoutError as e:
        log_error("evaluate_custom_json", e)
        raise HTTPException(status_code=504, detail=_error_detail(e))
    except Exception as e:
        log_error("evaluate_custom_json", e)
        raise HTTPException(status_code=500, detail=_error_detail(e))


@app.get(
    "/evaluate-all",
    response_model=BatchEvaluateResponse,
    summary="Run the full evaluation pipeline on all transcripts in the dataset",
)
@limiter.limit(RATE_LIMIT)
def evaluate_all(request: Request, model: Optional[str] = Query(default=None)):
    _require_openai_key()
    judge = JudgeLLM(model=model)
    results: list[EvaluationResult] = []
    errors: list[dict] = []
    all_transcripts = _get_transcripts()

    for transcript in all_transcripts:
        tid = transcript["transcript_id"]
        try:
            result = judge.evaluate(transcript)
            log_evaluation_result(result)
            results.append(result)
        except JudgeTimeoutError as e:
            log_error(f"evaluate_all:{tid}", e)
            errors.append({"transcript_id": tid, "error": _error_detail(e)})
        except Exception as e:
            log_error(f"evaluate_all:{tid}", e)
            errors.append({"transcript_id": tid, "error": _error_detail(e)})

    log_pipeline_summary(results)

    attempted = len(all_transcripts)
    total = len(results)
    passed = sum(1 for r in results if r.verdict == "PASS")
    failed = sum(1 for r in results if r.verdict == "FAIL")
    hard_failed = sum(1 for r in results if r.verdict == "HARD_FAIL")

    return BatchEvaluateResponse(
        attempted=attempted,
        total=total,
        passed=passed,
        failed=failed,
        hard_failed=hard_failed,
        pass_rate=round(passed / total, 3) if total else 0,
        results=results,
        errors=errors,
    )


@app.get("/audit-log", summary="Retrieve the structured audit log (last N entries)")
def get_audit_log(limit: int = Query(default=50, ge=1, le=500)):
    log_path = Path(__file__).parent.parent / "logs" / "evaluation_audit.jsonl"
    if not log_path.exists():
        return {"entries": []}
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    entries = [json.loads(line) for line in lines if line.strip()]
    return {"entries": entries[-limit:], "total_in_log": len(entries)}

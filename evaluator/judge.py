"""
Judge LLM: Orchestrates evaluation of a transcript against three criteria.
Uses prompts, validation, parsers, and core verdict logic.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

from evaluator.core import (
    EmpathyScore,
    EvaluationResult,
    Flag,
    GroundednessScore,
    MedicalSafetyScore,
    Verdict,
    compute_verdict,
)
from evaluator.parsers import JudgeParseError, parse_empathy, parse_groundedness, parse_safety
from evaluator.prompts import (
    EMPATHY_SYSTEM,
    GROUNDEDNESS_SYSTEM_TEMPLATE,
    SAFETY_SYSTEM,
)
from evaluator.validation import validate_transcript
from evaluator.logger import (
    log_evaluation_start,
    log_step_begin,
    log_step_safety_result,
    log_step_empathy_result,
    log_step_groundedness_result,
    log_step_verdict,
)


class JudgeTimeoutError(Exception):
    """Raised when the Judge LLM (OpenAI) request times out."""

    def __init__(self, message: str = "Judge LLM request timed out"):
        super().__init__(message)


class JudgeLLM:
    """
    Uses an LLM (default: gpt-4o) to evaluate agent transcripts against
    three criteria: Empathy, Groundedness, and Medical Safety.
    """

    def __init__(self, model: Optional[str] = None, timeout: Optional[float] = None):
        self.model = model or os.getenv("JUDGE_MODEL", "gpt-4o")
        _timeout = timeout if timeout is not None else float(os.getenv("OPENAI_TIMEOUT", "60.0"))
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=_timeout,
        )

    def _call(self, system: str, user_message: str, criterion: Optional[str] = None) -> dict:
        """Makes a single LLM call and parses the JSON response.
        Raises JudgeParseError if the response is empty or not valid JSON.
        Raises JudgeTimeoutError if the request times out.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            err_str = str(e).lower()
            err_type = type(e).__name__.lower()
            if "timeout" in err_type or "timeout" in err_str or "timed out" in err_str:
                raise JudgeTimeoutError(f"Judge LLM request timed out: {e}") from e
            raise
        raw = response.choices[0].message.content
        if raw is None or not raw.strip():
            raise JudgeParseError(
                "Judge LLM returned empty or null content",
                criterion=criterion,
                raw_preview=None,
            )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            preview = (raw.strip()[:200] + "…") if len(raw.strip()) > 200 else raw.strip()
            raise JudgeParseError(
                f"Judge LLM returned invalid JSON: {e}",
                criterion=criterion,
                raw_preview=preview,
            ) from e

    def _format_transcript(self, turns: list[dict]) -> str:
        """Formats conversation turns into a readable string for the judge."""
        lines = []
        for turn in turns:
            role = turn["role"].upper()
            lines.append(f"{role}: {turn['content']}")
        return "\n\n".join(lines)

    def _build_groundedness_system(self, retrieved_chunks: list[dict]) -> str:
        """
        Builds the groundedness system prompt using only the retrieved chunks for this transcript.
        If no chunks are provided, the judge is told that any factual claim is unfaithful.
        """
        if not retrieved_chunks:
            context_str = "[NO CHUNKS WERE RETRIEVED — any factual health claim the agent makes is unfaithful]"
        else:
            context_str = "\n\n".join(
                f"[{chunk.get('chunk_id', 'N/A')}] (score: {chunk.get('retrieval_score', 'N/A')})\n"
                f"{chunk.get('text', '')}\n"
                f"Source: {chunk.get('source', 'N/A')}"
                for chunk in retrieved_chunks
            )
        return GROUNDEDNESS_SYSTEM_TEMPLATE.format(retrieved_context=context_str)

    def evaluate(self, transcript: dict) -> EvaluationResult:
        """
        Evaluates a single transcript against all three criteria.

        Args:
            transcript: A dict with transcript_id, title (optional), and turns.

        Returns:
            A fully populated EvaluationResult.

        Raises:
            ValueError: If transcript is invalid.
            JudgeParseError: If the judge LLM returns invalid JSON or shape.
        """
        validate_transcript(transcript)
        transcript_id = transcript["transcript_id"]
        title = transcript.get("title", transcript_id)
        turns = transcript["turns"]
        formatted = self._format_transcript(turns)
        user_message = f"Evaluate the following conversation:\n\n{formatted}"

        log_evaluation_start(transcript_id, title)

        # 1. Medical Safety — hard gate
        log_step_begin(1, "Medical Safety", "Evaluating agent response for medical safety...")
        safety_raw = self._call(SAFETY_SYSTEM, user_message, criterion="Medical Safety")
        medical_safety = parse_safety(safety_raw, criterion="Medical Safety")
        log_step_safety_result(medical_safety.safe, medical_safety.reasoning)

        if not medical_safety.safe:
            verdict_result = EvaluationResult(
                transcript_id=transcript_id,
                title=title,
                empathy=EmpathyScore(level="E0", reasoning="Not evaluated — medical safety violation detected.", passed=False),
                groundedness=GroundednessScore(level="G0", reasoning="Not evaluated — medical safety violation detected.", passed=False),
                medical_safety=medical_safety,
                flags=[Flag.MEDICAL_SAFETY_VIOLATION],
                verdict=Verdict.HARD_FAIL,
                model_used=self.model,
                evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            return verdict_result

        # 2. Empathy
        log_step_begin(2, "Empathy", "Evaluating empathetic quality of agent response...")
        empathy_raw = self._call(EMPATHY_SYSTEM, user_message, criterion="Empathy")
        empathy = parse_empathy(empathy_raw, criterion="Empathy")
        log_step_empathy_result(empathy.level, empathy.passed)

        # 3. Groundedness (use retrieved chunks when provided, else "no context" placeholder)
        log_step_begin(3, "Groundedness", "Evaluating groundedness against knowledge base...")
        retrieved_chunks = transcript.get("retrieved_chunks") or []
        groundedness_system = self._build_groundedness_system(retrieved_chunks)
        groundedness_raw = self._call(groundedness_system, user_message, criterion="Groundedness")
        groundedness = parse_groundedness(groundedness_raw, criterion="Groundedness")
        log_step_groundedness_result(
            groundedness.level,
            groundedness.passed,
            groundedness.referenced_guidelines,
            groundedness.hallucinated_claims,
        )

        # 4. Verdict
        log_step_begin(4, "Verdict", "Computing final verdict from empathy and groundedness...")
        verdict, flags = compute_verdict(empathy, groundedness)
        log_step_verdict(verdict, flags)

        return EvaluationResult(
            transcript_id=transcript_id,
            title=title,
            empathy=empathy,
            groundedness=groundedness,
            medical_safety=medical_safety,
            flags=flags,
            verdict=verdict,
            model_used=self.model,
            evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

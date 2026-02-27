"""
Compare evaluation results to expected verdict and flags from transcript metadata.
Used to validate that the judge output matches the dataset's expected outcomes.
"""

from typing import Any

from evaluator.core import EvaluationResult


def check_expected(result: EvaluationResult, transcript: dict) -> tuple[bool, list[str]]:
    """
    Compare the evaluation result to the transcript's expected_verdict and expected_flags.

    Args:
        result: The EvaluationResult from the judge.
        transcript: The transcript dict, optionally containing expected_verdict and expected_flags.

    Returns:
        (matched, messages): matched is True only if both verdict and flags match;
        messages is a list of mismatch descriptions (empty if matched).
    """
    messages: list[str] = []
    expected_verdict = transcript.get("expected_verdict")
    expected_flags = transcript.get("expected_flags", [])

    if expected_verdict is None and not expected_flags:
        return True, []  # No expectations defined

    # Normalize: result.verdict is Verdict enum, compare by value
    actual_verdict = result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict)
    actual_flags = sorted(f.value if hasattr(f, "value") else str(f) for f in result.flags)
    expected_flags_normalized = sorted(expected_flags) if expected_flags else []

    if expected_verdict is not None and actual_verdict != expected_verdict:
        messages.append(f"verdict: expected {expected_verdict}, got {actual_verdict}")

    if expected_flags is not None and actual_flags != expected_flags_normalized:
        messages.append(f"flags: expected {expected_flags_normalized}, got {actual_flags}")

    return len(messages) == 0, messages

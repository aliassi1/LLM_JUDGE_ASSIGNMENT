"""Core domain: score models, verdicts, flags, and verdict computation."""

from .criteria import (
    EMPATHY_LEVEL_LABELS,
    EMPATHY_PASS_LEVELS,
    GROUNDEDNESS_LEVEL_LABELS,
    GROUNDEDNESS_PASS_LEVELS,
    EmpathyScore,
    EvaluationResult,
    Flag,
    GroundednessScore,
    MedicalSafetyScore,
    Verdict,
    compute_verdict,
)

__all__ = [
    "EMPATHY_LEVEL_LABELS",
    "EMPATHY_PASS_LEVELS",
    "GROUNDEDNESS_LEVEL_LABELS",
    "GROUNDEDNESS_PASS_LEVELS",
    "EmpathyScore",
    "EvaluationResult",
    "Flag",
    "GroundednessScore",
    "MedicalSafetyScore",
    "Verdict",
    "compute_verdict",
]

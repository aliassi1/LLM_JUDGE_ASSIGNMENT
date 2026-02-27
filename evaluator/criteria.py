"""
Backward-compatible re-export of criteria from evaluator.core.
Prefer: from evaluator.core import ...
"""

from evaluator.core import (
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

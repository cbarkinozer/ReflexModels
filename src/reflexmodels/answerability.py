"""Metrics for the separate V1 answerability / insufficient-evidence signal."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .metrics import classification_metrics


def answerability_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    """Evaluate P(answerable), where 1 means supplied options are sufficient."""
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("answerability labels must be zero (insufficient) or one (answerable)")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("answerability probabilities must be finite values between zero and one")
    rows = [[1.0 - float(value), float(value)] for value in probabilities]
    metrics = classification_metrics(labels, rows)
    predictions = [int(value >= 0.5) for value in probabilities]
    true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {**metrics, "answerable_precision": precision, "answerable_recall": recall}


def selective_accuracy(
    correct: Sequence[bool], answerable_probabilities: Sequence[float], *, threshold: float
) -> dict[str, float]:
    """Measure accepted-decision coverage and accuracy at a fixed validation threshold."""
    if not correct or len(correct) != len(answerable_probabilities):
        raise ValueError("correct and probabilities must be non-empty and have equal length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    accepted = [is_correct for is_correct, probability in zip(correct, answerable_probabilities) if probability >= threshold]
    return {
        "coverage": len(accepted) / len(correct),
        "selective_accuracy": sum(accepted) / len(accepted) if accepted else 0.0,
    }


def select_answerability_threshold(
    correct: Sequence[bool], answerable_probabilities: Sequence[float], *, min_coverage: float = 0.0
) -> dict[str, float]:
    """Choose a V1 fallback threshold on validation data only.

    It maximizes selective accuracy while requiring the requested validation
    coverage. Ties retain more requests, then choose the lower threshold.
    Callers must report the selected threshold separately and must not call
    this helper on frozen evaluation or test data.
    """
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between zero and one")
    if not correct or len(correct) != len(answerable_probabilities):
        raise ValueError("correct and probabilities must be non-empty and have equal length")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 for value in answerable_probabilities):
        raise ValueError("answerability probabilities must be finite values between zero and one")
    candidates = sorted({float(value) for value in answerable_probabilities})
    eligible = []
    for threshold in candidates:
        result = selective_accuracy(correct, answerable_probabilities, threshold=threshold)
        if result["coverage"] >= min_coverage:
            eligible.append({"threshold": threshold, **result})
    if not eligible:
        raise ValueError("no threshold meets min_coverage")
    return max(eligible, key=lambda item: (item["selective_accuracy"], item["coverage"], -item["threshold"]))

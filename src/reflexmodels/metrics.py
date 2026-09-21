"""Dependency-free metrics with explicit multiclass probability semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _validate(labels: Sequence[int], probabilities: Sequence[Sequence[float]]) -> int:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    class_count = len(probabilities[0])
    if class_count < 2:
        raise ValueError("each probability row must contain at least two classes")
    for label, row in zip(labels, probabilities):
        if not isinstance(label, int) or not 0 <= label < class_count:
            raise ValueError("labels must be valid zero-based class indices")
        if len(row) != class_count or any(not 0.0 <= value <= 1.0 for value in row):
            raise ValueError("probability rows must have equal valid probabilities")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("each probability row must sum to one")
    return class_count


def classification_metrics(
    labels: Sequence[int], probabilities: Sequence[Sequence[float]], *, ece_bins: int = 15
) -> dict[str, float]:
    """Return accuracy, macro-F1, NLL, multiclass Brier, and top-label ECE.

    Inputs are conditional Choice probabilities. Callers must evaluate Binary
    and answerability independently when their probability semantics differ.
    """
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive")
    class_count = _validate(labels, probabilities)
    predictions = [max(range(class_count), key=row.__getitem__) for row in probabilities]
    total = len(labels)
    accuracy = sum(actual == predicted for actual, predicted in zip(labels, predictions)) / total

    f1_values = []
    for class_index in range(class_count):
        true_positive = sum(
            actual == class_index and predicted == class_index
            for actual, predicted in zip(labels, predictions)
        )
        false_positive = sum(
            actual != class_index and predicted == class_index
            for actual, predicted in zip(labels, predictions)
        )
        false_negative = sum(
            actual == class_index and predicted != class_index
            for actual, predicted in zip(labels, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    epsilon = 1e-12
    nll = -sum(math.log(max(row[label], epsilon)) for label, row in zip(labels, probabilities)) / total
    brier = sum(
        sum((probability - float(index == label)) ** 2 for index, probability in enumerate(row))
        for label, row in zip(labels, probabilities)
    ) / total

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(ece_bins)]
    for actual, predicted, row in zip(labels, predictions, probabilities):
        confidence = row[predicted]
        bin_index = min(int(confidence * ece_bins), ece_bins - 1)
        bins[bin_index].append((confidence, actual == predicted))
    ece = sum(
        len(bucket) / total
        * abs(
            sum(confidence for confidence, _ in bucket) / len(bucket)
            - sum(correct for _, correct in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / class_count,
        "nll": nll,
        "brier": brier,
        "ece": ece,
    }

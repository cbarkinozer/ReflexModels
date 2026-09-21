"""Post-hoc temperature scaling and reliability data for Choice probabilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .metrics import _validate


def temperature_scale(probabilities: Sequence[Sequence[float]], temperature: float) -> list[list[float]]:
    """Apply temperature scaling to a normalized probability distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    _validate([0] * len(probabilities), probabilities)
    scaled = []
    for row in probabilities:
        powered = [max(value, 1e-12) ** (1.0 / temperature) for value in row]
        normalizer = sum(powered)
        scaled.append([value / normalizer for value in powered])
    return scaled


def nll(labels: Sequence[int], probabilities: Sequence[Sequence[float]]) -> float:
    _validate(labels, probabilities)
    return -sum(math.log(max(row[label], 1e-12)) for label, row in zip(labels, probabilities)) / len(labels)


def fit_temperature(labels: Sequence[int], probabilities: Sequence[Sequence[float]], *, iterations: int = 80) -> float:
    """Fit one positive scalar on validation data using golden-section search."""
    _validate(labels, probabilities)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    lower, upper = math.log(0.05), math.log(10.0)
    ratio = (math.sqrt(5) - 1) / 2
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    for _ in range(iterations):
        left_loss = nll(labels, temperature_scale(probabilities, math.exp(left)))
        right_loss = nll(labels, temperature_scale(probabilities, math.exp(right)))
        if left_loss <= right_loss:
            upper, right = right, left
            left = upper - ratio * (upper - lower)
        else:
            lower, left = left, right
            right = lower + ratio * (upper - lower)
    return math.exp((lower + upper) / 2)


def reliability_bins(
    labels: Sequence[int], probabilities: Sequence[Sequence[float]], *, bins: int = 15
) -> list[dict[str, float | int]]:
    """Return top-label reliability-diagram values without plotting dependencies."""
    if bins < 1:
        raise ValueError("bins must be positive")
    class_count = _validate(labels, probabilities)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for label, row in zip(labels, probabilities):
        prediction = max(range(class_count), key=row.__getitem__)
        confidence = row[prediction]
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, prediction == label))
    return [
        {
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": len(bucket),
            "confidence": sum(value for value, _ in bucket) / len(bucket),
            "accuracy": sum(correct for _, correct in bucket) / len(bucket),
        }
        for index, bucket in enumerate(buckets) if bucket
    ]

"""Deterministic confidence gates for a larger-model fallback path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def fallback_gate(
    response: Mapping[str, Any], *, min_confidence: float, min_margin: float = 0.0,
    max_normalized_entropy: float = 1.0, answerability_probability: float | None = None,
    min_answerability: float | None = None,
) -> dict[str, Any]:
    """Return deterministic routing metadata without changing typed output.

    The small decision model always returns its bounded candidate distribution.
    Consumers may then route low-confidence, ambiguous, or insufficient-
    evidence cases to a larger model. Answerability is supplied separately so
    V1 can add it without changing the bounded response wire format.
    Thresholds must be selected on validation data, never test data.
    """
    if not 0 < min_confidence <= 1 or not 0 <= min_margin <= 1 or not 0 <= max_normalized_entropy <= 1:
        raise ValueError("gate thresholds must be within probability bounds")
    if (answerability_probability is None) != (min_answerability is None):
        raise ValueError("answerability probability and threshold must be supplied together")
    if answerability_probability is not None:
        if (not isinstance(answerability_probability, (int, float)) or isinstance(answerability_probability, bool)
                or not math.isfinite(answerability_probability) or not 0 <= answerability_probability <= 1
                or not 0 < min_answerability <= 1):
            raise ValueError("answerability values must be within probability bounds")
    probabilities = response.get("option_probabilities")
    selected = response.get("selected_option")
    if not isinstance(probabilities, Mapping) or len(probabilities) < 2 or selected not in probabilities:
        raise ValueError("response must contain at least two candidate probabilities and a selected option")
    values = list(probabilities.values())
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("response probabilities must be finite values between zero and one")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("response probabilities must sum to one")
    ranked = sorted((float(value) for value in values), reverse=True)
    confidence = float(probabilities[selected])
    margin = ranked[0] - ranked[1]
    entropy = -sum(value * math.log(value) for value in values if value) / math.log(len(values))
    reasons = []
    if confidence < min_confidence:
        reasons.append("low_confidence")
    if margin < min_margin:
        reasons.append("low_margin")
    if entropy > max_normalized_entropy:
        reasons.append("high_entropy")
    if answerability_probability is not None and answerability_probability < min_answerability:
        reasons.append("insufficient_evidence")
    return {
        "accepted": not reasons,
        "fallback_required": bool(reasons),
        "reason_codes": reasons,
        "confidence": confidence,
        "margin": margin,
        "normalized_entropy": entropy,
        "answerability_probability": answerability_probability,
    }

"""Strict JSON wire format for constrained Reflex decisions.

The protocol purposefully has no free-text output field. A model may only emit
a selected candidate that appeared in the request plus its normalized scores.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from .types import Binary, Choice, Decision, Ordinal, SharedStateDecisions

VERSION = "reflex.decision.v1"


def request_to_dict(request: Binary | Choice | Ordinal, *, request_id: str) -> dict[str, Any]:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    base: dict[str, Any] = {
        "protocol": VERSION,
        "request_id": request_id,
        "language": request.language,
        "state": request.state,
        "question": request.question,
    }
    if isinstance(request, Choice):
        return {**base, "decision_type": "choice", "options": dict(request.options)}
    if isinstance(request, Binary):
        return {**base, "decision_type": "binary", "options": decision_candidates(request)}
    return {**base, "decision_type": "ordinal", "options": {level: level for level in request.levels}}


def request_to_json(request: Binary | Choice | Ordinal, *, request_id: str) -> str:
    return json.dumps(request_to_dict(request, request_id=request_id), ensure_ascii=False, separators=(",", ":"))


def batch_request_to_dict(batch: SharedStateDecisions, *, request_id: str) -> dict[str, Any]:
    """Encode independent decisions once with their common state text."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    decisions = []
    for decision_id, decision in batch.decisions.items():
        encoded = request_to_dict(decision, request_id=request_id)
        item = {key: value for key, value in encoded.items() if key not in {"protocol", "request_id", "language", "state"}}
        decisions.append({**item, "decision_id": decision_id})
    return {"protocol": VERSION, "request_id": request_id, "language": batch.language, "state": batch.state, "decisions": decisions}


def decision_candidates(decision: Decision) -> dict[str, str]:
    if isinstance(decision, Choice):
        return dict(decision.options)
    if isinstance(decision, Binary):
        return {"yes": "Yes", "no": "No"} if decision.language == "en" else {"yes": "Evet", "no": "Hayır"}
    return {level: level for level in decision.levels}


def decision_response(decision: Decision, *, request_id: str, probabilities: Mapping[str, float]) -> dict[str, Any]:
    """Return a typed response whose selected value must be a supplied candidate."""
    candidates = decision_candidates(decision)
    if set(probabilities) != set(candidates):
        raise ValueError("probabilities must contain exactly the supplied candidate names")
    values = {}
    for name, value in probabilities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("probabilities must be finite values between zero and one")
        values[name] = float(value)
    if not math.isclose(sum(values.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")
    decision_type = "choice" if isinstance(decision, Choice) else "binary" if isinstance(decision, Binary) else "ordinal"
    selected = max(values, key=values.__getitem__)
    return {"protocol": VERSION, "request_id": request_id, "decision_type": decision_type, "language": decision.language, "selected_option": selected, "confidence": values[selected], "option_probabilities": values}


def choice_response(
    request: Choice, *, request_id: str, probabilities: Mapping[str, float]
) -> dict[str, Any]:
    """Build a schema-valid choice response with no unconstrained text output."""
    return decision_response(request, request_id=request_id, probabilities=probabilities)


def response_to_json(response: Mapping[str, Any]) -> str:
    """Encode a response after validating only the constrained output contract."""
    required = {"protocol", "request_id", "decision_type", "language", "selected_option", "confidence", "option_probabilities"}
    if set(response) != required or response.get("protocol") != VERSION:
        raise ValueError("response does not match reflex.decision.v1")
    if response["selected_option"] not in response["option_probabilities"]:
        raise ValueError("selected_option must occur in option_probabilities")
    return json.dumps(dict(response), ensure_ascii=False, separators=(",", ":"))

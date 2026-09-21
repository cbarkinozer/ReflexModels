"""Model-agnostic isolated-branch scoring for the Reflex correctness oracle."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .protocol import decision_response
from .serialization import duplicated_decision_branches
from .types import Decision, SharedStateDecisions

BranchScorer = Callable[[str], float]


def softmax(scores: dict[str, float]) -> dict[str, float]:
    """Turn one finite independent score per candidate into probabilities."""
    if not scores or any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores.values()):
        raise ValueError("scores must be non-empty finite numeric values")
    offset = max(scores.values())
    weights = {name: math.exp(score - offset) for name, score in scores.items()}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def score_decision(request: Decision, *, request_id: str, scorer: BranchScorer) -> dict:
    """Score every candidate in a separate branch, then emit a typed result."""
    branches = duplicated_decision_branches(request)
    return decision_response(request, request_id=request_id, probabilities=softmax({name: scorer(branch) for name, branch in branches.items()}))


def score_shared_state(
    batch: SharedStateDecisions, *, request_id: str, scorer: BranchScorer, workers: int = 1
) -> dict[str, dict]:
    """Evaluate independent decisions concurrently when the scorer is thread-safe."""
    if workers < 1:
        raise ValueError("workers must be positive")
    def task(item: tuple[str, Decision]) -> tuple[str, dict]:
        decision_id, decision = item
        return decision_id, score_decision(decision, request_id=request_id, scorer=scorer)
    if workers == 1:
        pairs = map(task, batch.decisions.items())
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pairs = list(executor.map(task, batch.decisions.items()))
    return dict(pairs)

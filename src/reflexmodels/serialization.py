"""Canonical independent-branch serialization for the duplicated oracle."""

from __future__ import annotations

from .protocol import decision_candidates
from .types import Choice, Decision

STATE = "<STATE>"
QUESTION = "<QUESTION>"
OPTION = "<OPTION>"
DECISION = "<DECISION>"


def duplicated_choice_branches(request: Choice) -> dict[str, str]:
    """Return one causally independent prompt per runtime-defined option.

    Each branch has identical state/question text and exactly one option. This
    is the correctness oracle for later packed branch-isolated execution.
    """
    prefix = f"{STATE}\n{request.state}\n{QUESTION}\n{request.question}\n"
    return {
        name: f"{prefix}{OPTION}\n{description}\n{DECISION}"
        for name, description in request.options.items()
    }


def duplicated_decision_branches(request: Decision) -> dict[str, str]:
    """Return isolated branches for choice, binary, or ordinal decisions."""
    prefix = f"{STATE}\n{request.state}\n{QUESTION}\n{request.question}\n"
    return {
        name: f"{prefix}{OPTION}\n{description}\n{DECISION}"
        for name, description in decision_candidates(request).items()
    }

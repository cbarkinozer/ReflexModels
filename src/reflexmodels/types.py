"""Typed, model-agnostic decision request contracts.

These types define the public semantic interface only. They do not prescribe a
model architecture or inference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Binary:
    """A semantic yes/no decision with absolute probability semantics."""

    state: str
    question: str
    language: str = "tr"

    def __post_init__(self) -> None:
        _text(self.state, "state")
        _text(self.question, "question")
        _text(self.language, "language")


@dataclass(frozen=True, slots=True)
class Choice:
    """A relative choice among runtime-defined, natural-language options."""

    state: str
    question: str
    options: Mapping[str, str]
    language: str = "tr"

    def __post_init__(self) -> None:
        _text(self.state, "state")
        _text(self.question, "question")
        _text(self.language, "language")
        if not isinstance(self.options, Mapping) or len(self.options) < 2:
            raise ValueError("options must contain at least two named options")
        normalized = {}
        for name, description in self.options.items():
            normalized[_text(name, "option name")] = _text(description, "option description")
        if len(normalized) != len(self.options):
            raise ValueError("option names must be unique")
        object.__setattr__(self, "options", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class Ordinal:
    """An ordered decision; levels are not interchangeable nominal labels."""

    state: str
    question: str
    levels: tuple[str, ...]
    language: str = "tr"

    def __post_init__(self) -> None:
        _text(self.state, "state")
        _text(self.question, "question")
        _text(self.language, "language")
        if len(self.levels) < 2:
            raise ValueError("levels must contain at least two ordered values")
        normalized = tuple(_text(level, "ordinal level") for level in self.levels)
        if len(set(normalized)) != len(normalized):
            raise ValueError("ordinal levels must be unique")
        object.__setattr__(self, "levels", normalized)


Decision = Binary | Choice | Ordinal


@dataclass(frozen=True, slots=True)
class SharedStateDecisions:
    """Independent typed decisions that share one immutable input state."""

    state: str
    decisions: Mapping[str, Decision]
    language: str = "tr"

    def __post_init__(self) -> None:
        _text(self.state, "state")
        _text(self.language, "language")
        if not isinstance(self.decisions, Mapping) or not self.decisions:
            raise ValueError("decisions must contain at least one named decision")
        normalized: dict[str, Decision] = {}
        for name, decision in self.decisions.items():
            _text(name, "decision name")
            if not isinstance(decision, (Binary, Choice, Ordinal)):
                raise ValueError("each decision must be Binary, Choice, or Ordinal")
            if decision.state != self.state or decision.language != self.language:
                raise ValueError("each decision must use the shared state and language")
            normalized[name] = decision
        object.__setattr__(self, "decisions", MappingProxyType(normalized))

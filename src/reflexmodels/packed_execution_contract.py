"""Correctness contract between the V0 duplicated-branch oracle and any V2
shared-prefix / packed executor.

PLAN.md requires: "Prove packed results match the duplicated-branch oracle
before using the optimization in evaluation." This module does not implement
packed execution; it is the reusable check that requirement demands. Future
V2 code plugs a candidate executor into ``assert_packed_matches_oracle`` and
must pass on ``reference_batches()`` before any packed result is trusted for
evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from .scoring import BranchScorer, score_shared_state
from .types import Binary, Choice, Ordinal, SharedStateDecisions

PackedExecutor = Callable[..., dict[str, dict[str, Any]]]

# Two named tolerance presets. Exact-arithmetic scorers (pure Python, or a
# float32/float64 model run the same way on both sides) should clear the
# tight default. A real packed executor with a BF16/quantized numeric kernel
# will not match a float64 oracle to 1e-9; callers must pick and record an
# explicit, looser tolerance for that comparison rather than silently
# loosening the default everywhere.
EXACT_TOLERANCE = {"rel_tol": 1e-9, "abs_tol": 1e-9}
QUANTIZED_TOLERANCE = {"rel_tol": 1e-2, "abs_tol": 1e-3}


def assert_packed_matches_oracle(
    packed_executor: PackedExecutor,
    batch: SharedStateDecisions,
    *,
    request_id: str,
    scorer: BranchScorer,
    rel_tol: float = EXACT_TOLERANCE["rel_tol"],
    abs_tol: float = EXACT_TOLERANCE["abs_tol"],
) -> None:
    """Raise AssertionError if ``packed_executor`` disagrees with the oracle.

    Both implementations receive the same ``scorer`` (branch text -> score),
    so any disagreement reflects the packed implementation itself, not the
    underlying model. Every decision's selected option, decision type, and
    full candidate probability distribution must match within tolerance.
    ``rel_tol``/``abs_tol`` default to :data:`EXACT_TOLERANCE`; pass
    :data:`QUANTIZED_TOLERANCE` (or a run-specific value, recorded alongside
    the result) when the packed side runs a lower-precision numeric kernel.
    """
    oracle = score_shared_state(batch, request_id=request_id, scorer=scorer)
    packed = packed_executor(batch, request_id=request_id, scorer=scorer)
    if set(packed) != set(oracle):
        raise AssertionError(
            f"packed executor returned decision ids {sorted(packed)}, oracle returned {sorted(oracle)}"
        )
    for decision_id, expected in oracle.items():
        actual = packed[decision_id]
        for key in ("protocol", "request_id", "decision_type", "language", "selected_option"):
            if actual.get(key) != expected[key]:
                raise AssertionError(
                    f"{decision_id}: packed {key}={actual.get(key)!r} != oracle {key}={expected[key]!r}"
                )
        expected_probabilities = expected["option_probabilities"]
        actual_probabilities = actual.get("option_probabilities", {})
        if set(actual_probabilities) != set(expected_probabilities):
            raise AssertionError(f"{decision_id}: packed and oracle candidate sets differ")
        for option, probability in expected_probabilities.items():
            packed_probability = actual_probabilities[option]
            if not math.isclose(packed_probability, probability, rel_tol=rel_tol, abs_tol=abs_tol):
                raise AssertionError(
                    f"{decision_id}: option {option!r} probability {packed_probability} != oracle {probability}"
                )


def reference_batches() -> list[SharedStateDecisions]:
    """Fixed batches a candidate V2 packed executor must reproduce exactly.

    Beyond basic Binary/Choice/Ordinal coverage, this deliberately includes
    the cases most likely to break a packed/shared-prefix implementation
    that a duplicated-branch oracle cannot get wrong by construction:
    option-key order, two options sharing identical text under different
    ids (a real scorer can tie on these), long state text, full Turkish
    orthography (including the dotted/dotless I distinction), and multiple
    decisions in one batch with very different candidate counts.
    """
    state_a = "Kullanıcı config.yaml dosyasını incelemek istiyor."
    state_b = "Kullanıcı hava durumunu öğrenmek istiyor ama konum belirtilmemiş."
    state_turkish = "İstanbul'da yağış ihtimali güçlü; İzmir'de öğleden sonra açılıyor, çiğdem çiçeği açtı."
    state_long = (
        "Kullanıcı önce config.yaml, sonra .env, sonra docker-compose.yml dosyalarını sırayla inceledi. "
        * 40
    ) + "Şimdi hangi dosyanın değiştirileceğine karar vermesi gerekiyor."
    return [
        SharedStateDecisions(state_a, {
            "tool": Choice(
                state_a, "Hangi araç kullanılmalı?",
                {"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"},
            ),
        }),
        SharedStateDecisions(state_a, {
            "tool": Choice(
                state_a, "Hangi araç kullanılmalı?",
                {"filesystem": "Yerel dosyaları oku", "web": "İnternette ara", "shell": "Komut çalıştır"},
            ),
            "continue": Binary(state_a, "Devam edilsin mi?"),
        }),
        SharedStateDecisions(state_b, {
            "confidence": Ordinal(state_b, "Ne kadar eminsin?", ("düşük", "orta", "yüksek")),
            "ask_location": Binary(state_b, "Konum sorulmalı mı?"),
        }),
        # Same two options, opposite key insertion order: the contract must
        # not depend on dict iteration order.
        SharedStateDecisions(state_a, {
            "tool": Choice(
                state_a, "Hangi araç kullanılmalı?",
                {"web": "İnternette ara", "filesystem": "Yerel dosyaları oku"},
            ),
        }),
        # Two option ids with identical option text: only the id disambiguates.
        SharedStateDecisions(state_a, {
            "tool": Choice(
                state_a, "Hangi araç kullanılmalı?",
                {"tool_a": "Yerel dosyaları oku", "tool_b": "Yerel dosyaları oku"},
            ),
        }),
        SharedStateDecisions(state_turkish, {
            "forecast": Choice(
                state_turkish, "Hangi şehir seçilmeli?",
                {"istanbul": "İstanbul'u seç", "izmir": "İzmir'i seç"},
            ),
            "confident": Binary(state_turkish, "Tahmine güveniyor musun?"),
        }),
        SharedStateDecisions(state_long, {
            "file": Choice(
                state_long, "Hangi dosya değiştirilmeli?",
                {"config": "config.yaml", "env": ".env", "compose": "docker-compose.yml"},
            ),
        }),
        # Three decisions in one batch with very different candidate counts.
        SharedStateDecisions(state_b, {
            "yes_no": Binary(state_b, "Devam edilsin mi?"),
            "tool": Choice(
                state_b, "Hangi araç kullanılmalı?",
                {"filesystem": "Yerel dosyaları oku", "web": "İnternette ara", "shell": "Komut çalıştır"},
            ),
            "priority": Ordinal(
                state_b, "Öncelik nedir?",
                ("çok düşük", "düşük", "orta", "yüksek", "çok yüksek", "acil"),
            ),
        }),
    ]

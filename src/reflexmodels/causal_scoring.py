"""Non-generative causal-LM candidate scoring for Reflex V0.

Each option is scored as an independent continuation of the same state and
question. This is intentionally an unoptimized correctness oracle: it does
not let sibling options interact and does not generate text.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .protocol import decision_candidates, decision_response
from .serialization import OPTION, QUESTION, STATE
from .scoring import softmax
from .types import Decision


def branch_prefix(state: str, question: str) -> str:
    """Return the shared textual prefix used by every V0 option branch."""
    if not isinstance(state, str) or not state.strip() or not isinstance(question, str) or not question.strip():
        raise ValueError("state and question must be non-empty strings")
    return f"{STATE}\n{state}\n{QUESTION}\n{question}\n{OPTION}\n"


def candidate_sequences(tokenizer: Any, *, state: str, question: str, options: Mapping[str, str]) -> dict[str, tuple[list[int], list[int]]]:
    """Tokenize independent prefix/option pairs without cross-option context.

    The return value is ``option key -> (input ids, continuation ids)``.
    Keeping the continuation separately lets callers compute only its
    conditional log probability rather than scoring prompt tokens too.
    """
    if not isinstance(options, Mapping) or len(options) < 2:
        raise ValueError("options must contain at least two choices")
    prefix_ids = list(tokenizer.encode(branch_prefix(state, question), add_special_tokens=False))
    if not prefix_ids:
        raise ValueError("tokenizer produced an empty prefix")
    result: dict[str, tuple[list[int], list[int]]] = {}
    for key, text in options.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("option keys and texts must be non-empty strings")
        continuation_ids = list(tokenizer.encode(text, add_special_tokens=False))
        if not continuation_ids:
            raise ValueError(f"tokenizer produced an empty continuation for {key!r}")
        result[key] = (prefix_ids + continuation_ids, continuation_ids)
    return result


def option_logprobabilities(
    model: Any,
    tokenizer: Any,
    *,
    state: str,
    question: str,
    options: Mapping[str, str],
    length_normalize: bool = True,
) -> dict[str, float]:
    """Return independent causal-LM option scores, with no text generation.

    Scores are mean continuation log probabilities by default. Length
    normalization prevents longer option descriptions being penalized merely
    for containing more tokens. This is a V0 baseline score, not calibrated
    probability or an answerability estimate.
    """
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in model runs
        raise RuntimeError("option scoring requires PyTorch") from error

    sequences = candidate_sequences(tokenizer, state=state, question=question, options=options)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for key, (input_ids, continuation_ids) in sequences.items():
            values = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits = model(input_ids=values).logits[0]
            start = len(input_ids) - len(continuation_ids)
            token_log_probs = torch.log_softmax(logits[start - 1:-1], dim=-1)
            targets = values[0, start:]
            score = token_log_probs.gather(1, targets.unsqueeze(1)).sum().item()
            scores[key] = score / len(continuation_ids) if length_normalize else score
    model.train(was_training)
    return scores


def score_causal_decision(
    request: Decision,
    *,
    request_id: str,
    model: Any,
    tokenizer: Any,
    length_normalize: bool = True,
) -> dict:
    """Score a typed runtime decision with the V0 causal-LM oracle."""
    scores = option_logprobabilities(
        model,
        tokenizer,
        state=request.state,
        question=request.question,
        options=decision_candidates(request),
        length_normalize=length_normalize,
    )
    return decision_response(request, request_id=request_id, probabilities=softmax(scores))

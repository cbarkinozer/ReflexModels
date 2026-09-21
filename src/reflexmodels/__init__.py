"""Shared evaluation primitives for ReflexModels experiments."""

from .metrics import classification_metrics
from .types import Binary, Choice, Ordinal, SharedStateDecisions
from .calibration import fit_temperature, reliability_bins, temperature_scale
from .split_policy import apply_overlap_policy, normalized_text_hash
from .protocol import VERSION, batch_request_to_dict, choice_response, decision_response, request_to_dict, request_to_json, response_to_json
from .scoring import score_decision, score_shared_state, softmax
from .routing import fallback_gate
from .answerability import answerability_metrics, select_answerability_threshold, selective_accuracy
from .causal_scoring import branch_prefix, candidate_sequences, option_logprobabilities, score_causal_decision

__all__ = [
    "Binary", "Choice", "Ordinal", "SharedStateDecisions", "classification_metrics", "fit_temperature",
    "reliability_bins", "temperature_scale", "apply_overlap_policy", "normalized_text_hash",
    "VERSION", "batch_request_to_dict", "choice_response", "decision_response", "request_to_dict", "request_to_json", "response_to_json",
    "score_decision", "score_shared_state", "softmax",
    "fallback_gate",
    "answerability_metrics", "select_answerability_threshold", "selective_accuracy",
    "branch_prefix", "candidate_sequences", "option_logprobabilities", "score_causal_decision",
]

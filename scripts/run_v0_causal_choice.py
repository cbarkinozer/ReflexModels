"""Run Reflex V0 duplicated causal-LM scoring on one local typed request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexmodels.causal_scoring import score_causal_decision
from reflexmodels.protocol import response_to_json
from reflexmodels.types import Binary, Choice, Decision, Ordinal


def load_request(path: Path) -> tuple[str, Decision]:
    """Load a minimal wire request for a runtime-defined typed decision."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    required = {"request_id", "state", "question"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"request missing fields: {', '.join(sorted(missing))}")
    request_id = payload["request_id"]
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    language = payload.get("language", "tr")
    decision_type = payload.get("decision_type", "choice")
    if decision_type == "choice":
        if "options" not in payload:
            raise ValueError("choice request missing fields: options")
        return request_id, Choice(state=payload["state"], question=payload["question"], options=payload["options"], language=language)
    if decision_type == "binary":
        return request_id, Binary(state=payload["state"], question=payload["question"], language=language)
    if decision_type == "ordinal":
        if "levels" not in payload:
            raise ValueError("ordinal request missing fields: levels")
        levels = payload["levels"]
        if not isinstance(levels, list):
            raise ValueError("ordinal levels must be a JSON array")
        return request_id, Ordinal(state=payload["state"], question=payload["question"], levels=tuple(levels), language=language)
    raise ValueError("decision_type must be choice, binary, or ordinal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="Local causal-LM checkpoint directory")
    parser.add_argument("--request", required=True, type=Path, help="JSON choice request")
    parser.add_argument("--output", type=Path, help="Optional response JSON file")
    parser.add_argument("--sum-logprob", action="store_true", help="Do not normalize scores by option-token count")
    parser.add_argument("--num-threads", type=int, help="CPU threads for PyTorch")
    args = parser.parse_args()
    if args.num_threads is not None and args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    if not args.model_path.is_dir():
        raise ValueError("model-path must be an existing local checkpoint directory")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install compatible torch and transformers in .venv before running") from error

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    request_id, request = load_request(args.request)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
    response = score_causal_decision(
        request,
        request_id=request_id,
        model=model,
        tokenizer=tokenizer,
        length_normalize=not args.sum_logprob,
    )
    rendered = response_to_json(response)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

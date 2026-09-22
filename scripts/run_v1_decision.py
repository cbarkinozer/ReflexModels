"""Run one bounded V1 decision from a saved local backbone and decision heads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reflexmodels.protocol import decision_candidates, decision_response
from reflexmodels.routing import fallback_gate
from reflexmodels.scoring import softmax
from run_v0_causal_choice import load_request
from train_v1_decision_model import DecisionExample, DecisionModel, predict


def build_output(request: Any, request_id: str, scores: dict[str, float], p_answerable: float,
                 *, temperature: float, answerability_threshold: float) -> dict[str, Any]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration temperature must be positive and finite")
    if set(scores) != set(decision_candidates(request)):
        raise ValueError("model scores must cover exactly the supplied options")
    probabilities = softmax({name: score / temperature for name, score in scores.items()})
    decision = decision_response(request, request_id=request_id, probabilities=probabilities)
    routing = fallback_gate(
        decision, min_confidence=1e-12, answerability_probability=p_answerable,
        min_answerability=max(answerability_threshold, 1e-12),
    )
    return {"decision": decision, "answerability_probability": p_answerable, "routing": routing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="V1 training output directory")
    parser.add_argument("--request", required=True, type=Path, help="Typed JSON request")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    import torch
    from transformers import AutoModel, AutoTokenizer

    metadata = json.loads((args.model_dir / "train_metadata.json").read_text(encoding="utf-8"))
    backbone_path = args.model_dir / "backbone"
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True)
    model = DecisionModel(backbone)
    heads = torch.load(args.model_dir / "decision_heads.pt", map_location="cpu", weights_only=True)
    model.decision_head.load_state_dict(heads["decision_head"])
    model.answerability_head.load_state_dict(heads["answerability_head"])
    torch.set_num_threads(args.num_threads)
    request_id, request = load_request(args.request)
    example = DecisionExample(
        record_id=request_id, split="inference", state=request.state, question=request.question,
        options=tuple(decision_candidates(request).items()), answerable=False, correct_option=None,
    )
    scores, p_answerable = predict(
        model, [example], tokenizer, max_length=int(metadata["hyperparameters"]["max_length"]), batch_size=1
    )
    output = build_output(
        request, request_id, scores[0], p_answerable[0],
        temperature=float(metadata["calibration_temperature"]),
        answerability_threshold=float(metadata["answerability_threshold"]),
    )
    rendered = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

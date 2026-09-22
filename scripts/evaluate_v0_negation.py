"""Evaluate Reflex V0 on the frozen Turkish negation seed without generation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reflexmodels.causal_scoring import score_causal_decision
from reflexmodels.metrics import classification_metrics
from reflexmodels.types import Binary


def evaluate_records(records: list[dict], scorer) -> tuple[list[dict], dict]:
    predictions = []
    latencies = []
    for record in records:
        request = Binary(state=record["state"], question=record["question"], language="tr")
        started = time.perf_counter()
        response = scorer(request, record["id"])
        latencies.append((time.perf_counter() - started) * 1000)
        probabilities = response["option_probabilities"]
        predictions.append({
            "id": record["id"], "pair_id": record["pair_id"], "language": "tr",
            "label": int(record["label"]),
            "probabilities": [probabilities["no"], probabilities["yes"]],
            "selected_option": response["selected_option"],
        })
    metrics = classification_metrics(
        [row["label"] for row in predictions], [row["probabilities"] for row in predictions]
    )
    pairs: dict[str, list[dict]] = {}
    for row in predictions:
        pairs.setdefault(row["pair_id"], []).append(row)
    pair_accuracy = sum(
        all((row["selected_option"] == "yes") == bool(row["label"]) for row in pair)
        for pair in pairs.values()
    ) / len(pairs)
    metrics.update({
        "pair_accuracy": pair_accuracy,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p50": statistics.median(latencies),
    })
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    if not args.model_path.is_dir() or args.num_threads < 1:
        raise ValueError("a local model directory and positive thread count are required")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("validate_reflexbench_tr", ROOT / "scripts" / "validate_reflexbench_tr.py")
    validator = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(validator)
    dataset_path = ROOT / "benchmarks" / "reflexbench_tr" / "negation_seed.jsonl"
    records = validator.load_records(dataset_path)
    validator.validate(records)
    torch.set_num_threads(args.num_threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
    predictions, metrics = evaluate_records(
        records,
        lambda request, request_id: score_causal_decision(
            request, request_id=request_id, model=model, tokenizer=tokenizer
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    result = {"model_source": str(args.model_path), "dataset": str(dataset_path), "record_count": len(records),
              "method": "v0_independent_mean_option_logprob", "num_threads": args.num_threads, "metrics": metrics}
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

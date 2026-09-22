"""Evaluate a saved V1 model on a separate, frozen test JSONL.

Calibration temperature and answerability threshold are read from training
metadata. This command never fits or selects them on test examples.
"""

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

from reflexmodels.answerability import answerability_metrics, selective_accuracy
from reflexmodels.scoring import softmax
from reflexmodels.split_policy import normalized_text_hash
from train_v1_decision_model import DecisionExample, DecisionModel, VALIDATOR, decision_validation_metrics, predict


def load_test_examples(path: Path) -> tuple[list[DecisionExample], list[dict[str, Any]]]:
    records = VALIDATOR.load_records(path)
    VALIDATOR.validate(records, allowed_splits=frozenset({"test"}))
    examples = [
        DecisionExample(
            record_id=record["id"], split="test", state=record["state"], question=record["question"],
            options=tuple(record["options"].items()), answerable=record["answerable"],
            correct_option=record["correct_option"],
        )
        for record in records
    ]
    return examples, records


def reject_training_overlap(records: list[dict[str, Any]], fingerprint_path: Path) -> None:
    """Refuse exact normalized-state overlap with training or calibration data."""
    manifest = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if manifest.get("normalization") != "strip+casefold+sha256 of state text":
        raise ValueError("unknown training fingerprint normalization")
    seen = set(manifest["train_state_hashes"]) | set(manifest["validation_state_hashes"])
    duplicates = [record["id"] for record in records if normalized_text_hash(record["state"]) in seen]
    if duplicates:
        raise ValueError(f"test states overlap train/validation: {', '.join(duplicates[:10])}")


def evaluate_predictions(
    records: list[dict[str, Any]], score_maps: list[dict[str, float]],
    answerable_probabilities: list[float], *, temperature: float, threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute fixed-calibration metrics, with Turkish and English kept apart."""
    if not records or len(records) != len(score_maps) or len(records) != len(answerable_probabilities):
        raise ValueError("records and predictions must be non-empty and aligned")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= threshold <= 1:
        raise ValueError("temperature and threshold must be valid fixed calibration values")
    predictions = []
    for record, scores, p_answerable in zip(records, score_maps, answerable_probabilities):
        if record["split"] != "test":
            raise ValueError("evaluation accepts only split=test records")
        if set(scores) != set(record["options"]):
            raise ValueError(f"record {record['id']}: scores do not match supplied options")
        if not isinstance(p_answerable, (int, float)) or not math.isfinite(p_answerable) or not 0 <= p_answerable <= 1:
            raise ValueError(f"record {record['id']}: invalid answerability probability")
        if any(not math.isfinite(score) for score in scores.values()):
            raise ValueError(f"record {record['id']}: non-finite decision score")
        probabilities = softmax({name: score / temperature for name, score in scores.items()})
        selected = max(probabilities, key=probabilities.__getitem__)
        predictions.append({
            "id": record["id"], "language": record["language"], "origin": record["origin"],
            "answerable": record["answerable"], "correct_option": record["correct_option"],
            "selected_option": selected, "option_probabilities": probabilities,
            "answerability_probability": float(p_answerable), "accepted": p_answerable >= threshold,
        })
    by_language = {}
    for language in sorted({record["language"] for record in records}):
        indexes = [index for index, record in enumerate(records) if record["language"] == language]
        answerable_indexes = [index for index in indexes if records[index]["answerable"]]
        labels = [int(records[index]["answerable"]) for index in indexes]
        probabilities = [float(answerable_probabilities[index]) for index in indexes]
        correct = [
            bool(records[index]["answerable"] and predictions[index]["selected_option"] == records[index]["correct_option"])
            for index in indexes
        ]
        by_language[language] = {
            "record_count": len(indexes),
            "answerable_record_count": len(answerable_indexes),
            "decision": decision_validation_metrics(
                [score_maps[index] for index in answerable_indexes],
                [records[index]["correct_option"] for index in answerable_indexes],
                temperature=temperature,
            ) if answerable_indexes else None,
            "answerability": answerability_metrics(labels, probabilities),
            "coverage": selective_accuracy(correct, probabilities, threshold=threshold),
        }
    return {"calibration_temperature": temperature, "answerability_threshold": threshold,
            "by_language": by_language}, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Separate JSONL containing only split=test records")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_threads < 1:
        raise ValueError("batch-size and num-threads must be positive")
    import torch
    from transformers import AutoModel, AutoTokenizer

    metadata = json.loads((args.model_dir / "train_metadata.json").read_text(encoding="utf-8"))
    examples, records = load_test_examples(args.data)
    reject_training_overlap(records, args.model_dir / "training_data_fingerprints.json")
    backbone_path = args.model_dir / "backbone"
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    model = DecisionModel(AutoModel.from_pretrained(backbone_path, local_files_only=True))
    heads = torch.load(args.model_dir / "decision_heads.pt", map_location="cpu", weights_only=True)
    model.decision_head.load_state_dict(heads["decision_head"])
    model.answerability_head.load_state_dict(heads["answerability_head"])
    torch.set_num_threads(args.num_threads)
    score_maps, answerable_probabilities = predict(
        model, examples, tokenizer, max_length=int(metadata["hyperparameters"]["max_length"]),
        batch_size=args.batch_size,
    )
    metrics, predictions = evaluate_predictions(
        records, score_maps, answerable_probabilities,
        temperature=float(metadata["calibration_temperature"]),
        threshold=float(metadata["answerability_threshold"]),
    )
    metrics["model_dir"] = str(args.model_dir)
    metrics["test_data"] = str(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as destination:
        for prediction in predictions:
            destination.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

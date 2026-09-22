"""Evaluate a saved V1 model on a separate, frozen test JSONL.

Calibration temperature and answerability threshold are read from training
metadata. This command never fits or selects them on test examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reflexmodels.answerability import answerability_metrics, selective_accuracy
from reflexmodels.cpu_benchmark import measure_callable
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


def file_sha256(path: Path) -> str:
    """Hash a small saved artifact in streaming mode for run provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reproducibility_metadata(model_dir: Path, data_path: Path, *, torch_version: str) -> dict[str, Any]:
    """Identify the exact small artifacts and execution environment used."""
    backbone = model_dir / "backbone"
    required = {
        "training_metadata_sha256": model_dir / "train_metadata.json",
        "decision_heads_sha256": model_dir / "decision_heads.pt",
        "backbone_config_sha256": backbone / "config.json",
        "tokenizer_config_sha256": backbone / "tokenizer_config.json",
        "test_data_sha256": data_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f"required evaluation artifacts are missing: {', '.join(missing)}")
    return {
        **{key: file_sha256(path) for key, path in required.items()},
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch_version,
    }


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
        if "task_family" in record:
            predictions[-1]["task_family"] = record["task_family"]

    def metrics_for_indexes(indexes: list[int]) -> dict[str, Any]:
        answerable_indexes = [index for index in indexes if records[index]["answerable"]]
        labels = [int(records[index]["answerable"]) for index in indexes]
        probabilities = [float(answerable_probabilities[index]) for index in indexes]
        correct = [
            bool(records[index]["answerable"] and predictions[index]["selected_option"] == records[index]["correct_option"])
            for index in indexes
        ]
        return {
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

    by_language = {
        language: metrics_for_indexes([index for index, record in enumerate(records) if record["language"] == language])
        for language in sorted({record["language"] for record in records})
    }
    # Origin is mandatory in the V1 data contract. Always report it separately
    # so translated or synthetic evaluation examples cannot be mistaken for
    # native-Turkish performance.
    by_origin = {
        origin: metrics_for_indexes([index for index, record in enumerate(records) if record["origin"] == origin])
        for origin in sorted({record["origin"] for record in records})
    }
    # Task-family reporting is optional so generic V1 test files stay valid.
    # A benchmark that supplies it must do so for every record; otherwise
    # silently grouping a partial subset would make coverage misleading.
    task_families = [record.get("task_family") for record in records]
    if any(task_families) and not all(isinstance(family, str) and family.strip() for family in task_families):
        raise ValueError("task_family must be a non-empty string on every record when present")
    by_task_family = {
        family: metrics_for_indexes([index for index, record in enumerate(records) if record["task_family"] == family])
        for family in sorted(set(task_families))
    } if all(task_families) else None
    result = {"calibration_temperature": temperature, "answerability_threshold": threshold,
              "by_language": by_language, "by_origin": by_origin}
    if by_task_family is not None:
        result["by_task_family"] = by_task_family
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Separate JSONL containing only split=test records")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--latency-repetitions", type=int, default=0,
                        help="Optional single-request CPU measurements; zero disables timing")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_threads < 1 or args.latency_repetitions < 0:
        raise ValueError("batch-size and num-threads must be positive; latency-repetitions must be non-negative")
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
    metrics["reproducibility"] = reproducibility_metadata(args.model_dir, args.data, torch_version=torch.__version__)
    if args.latency_repetitions:
        single_example = examples[:1]
        metrics["cpu_single_request"] = measure_callable(
            lambda: predict(
                model, single_example, tokenizer, max_length=int(metadata["hyperparameters"]["max_length"]), batch_size=1
            ),
            warmup=1, repetitions=args.latency_repetitions,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as destination:
        for prediction in predictions:
            destination.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

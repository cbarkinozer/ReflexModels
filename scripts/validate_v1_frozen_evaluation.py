"""Recompute a saved frozen V1 evaluation's metrics from its predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexmodels.answerability import answerability_metrics, selective_accuracy


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_v1_decision_model", ROOT / "scripts" / "evaluate_v1_decision_model.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("predictions.jsonl contains no records")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError("prediction ids must be unique, non-empty strings")
    for row in rows:
        probabilities = row.get("option_probabilities")
        if not isinstance(probabilities, dict) or len(probabilities) < 2:
            raise ValueError(f"{row['id']}: option_probabilities must contain at least two candidates")
        if any(not isinstance(name, str) or not name or not isinstance(value, (int, float))
               or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1
               for name, value in probabilities.items()):
            raise ValueError(f"{row['id']}: option_probabilities must be finite probabilities")
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"{row['id']}: option_probabilities must sum to one")
        if row.get("selected_option") not in probabilities:
            raise ValueError(f"{row['id']}: selected_option is not a supplied candidate")
        probability = row.get("answerability_probability")
        if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{row['id']}: invalid answerability_probability")
        if not isinstance(row.get("answerable"), bool):
            raise ValueError(f"{row['id']}: answerable must be boolean")
        if row["answerable"] and row.get("correct_option") not in probabilities:
            raise ValueError(f"{row['id']}: correct_option is not a supplied candidate")
        if not row["answerable"] and row.get("correct_option") is not None:
            raise ValueError(f"{row['id']}: unanswerable prediction must have null correct_option")
    return rows


def _metrics_for_indexes(predictions: list[dict[str, Any]], indexes: list[int], *, threshold: float) -> dict[str, Any]:
    rows = [predictions[index] for index in indexes]
    answerable = [bool(row["answerable"]) for row in rows]
    answerability_probabilities = [float(row["answerability_probability"]) for row in rows]
    correct = [
        bool(row["answerable"] and row["selected_option"] == row["correct_option"])
        for row in rows
    ]
    answerable_rows = [row for row in rows if row["answerable"]]
    # Saved probabilities are already temperature calibrated. Their logarithms
    # are equivalent logits for recomputing conditional decision metrics.
    score_maps = [
        {name: math.log(max(float(value), 1e-300)) for name, value in row["option_probabilities"].items()}
        for row in answerable_rows
    ]
    return {
        "record_count": len(rows),
        "answerable_record_count": len(answerable_rows),
        "decision": EVALUATOR.decision_validation_metrics(
            score_maps, [row["correct_option"] for row in answerable_rows], temperature=1.0
        ) if answerable_rows else None,
        "answerability": answerability_metrics([int(value) for value in answerable], answerability_probabilities),
        "coverage": selective_accuracy(correct, answerability_probabilities, threshold=threshold),
    }


def _assert_close(actual: Any, expected: Any, *, path: str = "metrics") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path}: metric keys do not match predictions")
        for key in expected:
            _assert_close(actual[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"{path}: saved metric does not match predictions")
    elif actual != expected:
        raise ValueError(f"{path}: saved metric does not match predictions")


def validate(directory: Path) -> dict[str, int]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    predictions = load_predictions(directory / "predictions.jsonl")
    threshold = metrics.get("answerability_threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("metrics.json has no valid answerability_threshold")
    for row in predictions:
        if row.get("accepted") is not (float(row["answerability_probability"]) >= float(threshold)):
            raise ValueError(f"{row['id']}: accepted does not match saved answerability threshold")
    for key in ("by_language", "by_origin"):
        groups = metrics.get(key)
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"metrics.json has no non-empty {key}")
        field = key.removeprefix("by_")
        for group, saved_metrics in groups.items():
            indexes = [index for index, row in enumerate(predictions) if row.get(field) == group]
            _assert_close(saved_metrics, _metrics_for_indexes(predictions, indexes, threshold=float(threshold)), path=f"{key}.{group}")
    if "by_task_family" in metrics:
        groups = metrics["by_task_family"]
        for group, saved_metrics in groups.items():
            indexes = [index for index, row in enumerate(predictions) if row.get("task_family") == group]
            _assert_close(saved_metrics, _metrics_for_indexes(predictions, indexes, threshold=float(threshold)),
                          path=f"by_task_family.{group}")
    return {"predictions": len(predictions), "languages": len(metrics["by_language"]), "origins": len(metrics["by_origin"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory produced by evaluate_v1_decision_model.py")
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))


if __name__ == "__main__":
    main()

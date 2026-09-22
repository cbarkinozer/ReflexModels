"""Check a CPU baseline result against its saved predictions and checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reflexmodels.metrics import classification_metrics


def validate_result(directory: Path) -> dict[str, object]:
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    prediction_path = directory / "test_predictions.jsonl"
    predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = result["record_counts"]["test"]
    if len(predictions) != expected_count:
        raise ValueError(f"expected {expected_count} test predictions, found {len(predictions)}")
    ids = [record["id"] for record in predictions]
    if len(set(ids)) != len(ids):
        raise ValueError("test prediction ids are not unique")
    if any(record["language"] != result["language"] for record in predictions):
        raise ValueError("prediction language differs from result language")
    labels = [record["label"] for record in predictions]
    probabilities = [record["probabilities"] for record in predictions]
    recomputed = classification_metrics(labels, probabilities)
    for key, value in recomputed.items():
        if not math.isclose(value, result["test_metrics_temperature_calibrated"][key], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"calibrated metric {key} does not match saved predictions")
    checkpoint = Path(result["selected_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"selected checkpoint is missing: {checkpoint}")
    return {"predictions": len(predictions), "selected_epoch": result["selected_epoch"], "checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_result(args.directory), indent=2))


if __name__ == "__main__":
    main()

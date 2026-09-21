"""Evaluate JSONL Choice predictions without mixing Turkish and English metrics.

Input records require: ``id``, ``language`` (``tr`` or ``en``), integer
``label``, and a probability list named ``probabilities``. The output is a
single JSON document containing one metric block per language.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reflexmodels.metrics import classification_metrics


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_predictions(path: Path) -> dict[str, tuple[list[int], list[list[float]]]]:
    grouped: dict[str, tuple[list[int], list[list[float]]]] = {
        "tr": ([], []), "en": ([], [])
    }
    seen_ids: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record: dict[str, Any] = json.loads(line)
            identifier = record["id"]
            language = record["language"]
            label = record["label"]
            probabilities = record["probabilities"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{number}: invalid prediction record") from error
        if not isinstance(identifier, str) or identifier in seen_ids:
            raise ValueError(f"{path}:{number}: id must be unique")
        if language not in grouped:
            raise ValueError(f"{path}:{number}: language must be 'tr' or 'en'")
        if not isinstance(label, int) or not isinstance(probabilities, list):
            raise ValueError(f"{path}:{number}: invalid label or probabilities")
        labels, rows = grouped[language]
        labels.append(label)
        rows.append(probabilities)
        seen_ids.add(identifier)
    return {language: values for language, values in grouped.items() if values[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Model identifier and revision")
    parser.add_argument("--dataset", required=True, help="Dataset name/version/split")
    parser.add_argument("--config", required=True, help="Config path or identifier")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    grouped = load_predictions(args.predictions)
    if not grouped:
        raise ValueError("prediction file contains no records")
    result = {
        "run_id": f"evaluation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": args.model,
        "dataset": args.dataset,
        "config": args.config,
        "seed": args.seed,
        "task": "choice",
        "metrics_by_language": {
            language: classification_metrics(labels, probabilities)
            for language, (labels, probabilities) in grouped.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

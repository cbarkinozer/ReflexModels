"""Validate V1 decision-training JSONL records before they enter a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id", "language", "origin", "source", "split", "state", "question",
    "options", "answerable", "correct_option",
}
VALID_ORIGINS = {"native", "translated", "synthetic"}
VALID_SPLITS = {"train", "validation"}


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: record must be a JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: contains no records")
    return records


def validate(records: list[dict[str, Any]]) -> None:
    identifiers: set[str] = set()
    for number, record in enumerate(records, 1):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record {number}: missing fields: {', '.join(sorted(missing))}")
        identifier = record["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("every record id must be a unique, non-empty string")
        if record["language"] not in {"tr", "en"}:
            raise ValueError("language must be 'tr' or 'en'")
        if record["origin"] not in VALID_ORIGINS:
            raise ValueError("origin must be native, translated, or synthetic")
        if record["split"] not in VALID_SPLITS:
            raise ValueError("split must be train or validation")
        if not all(isinstance(record[field], str) and record[field].strip() for field in ("source", "state", "question")):
            raise ValueError("source, state, and question must be non-empty strings")
        options = record["options"]
        if not isinstance(options, dict) or len(options) < 2:
            raise ValueError("options must be an object with at least two choices")
        if not all(isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip() for key, value in options.items()):
            raise ValueError("option keys and texts must be non-empty strings")
        if not isinstance(record["answerable"], bool):
            raise ValueError("answerable must be boolean")
        correct_option = record["correct_option"]
        if record["answerable"] and correct_option not in options:
            raise ValueError("an answerable record must name a supplied correct_option")
        if not record["answerable"] and correct_option is not None:
            raise ValueError("an unanswerable record must set correct_option to null")
        identifiers.add(identifier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    records = load_records(args.path)
    validate(records)
    print(f"validated {len(records)} V1 decision records")


if __name__ == "__main__":
    main()

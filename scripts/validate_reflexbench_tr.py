"""Validate structural invariants of frozen ReflexBench-TR JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"id", "language", "origin", "task", "state", "question", "label", "pair_id"}
VALID_ORIGINS = {"native", "translated", "synthetic"}


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        missing = REQUIRED_FIELDS - record.keys() if isinstance(record, dict) else REQUIRED_FIELDS
        if missing:
            raise ValueError(f"{path}:{number}: missing fields: {', '.join(sorted(missing))}")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: contains no records")
    return records


def validate(records: list[dict[str, Any]]) -> None:
    identifiers: set[str] = set()
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record missing fields: {', '.join(sorted(missing))}")
        identifier = record["id"]
        if not isinstance(identifier, str) or identifier in identifiers:
            raise ValueError("every benchmark id must be a unique string")
        if record["language"] != "tr":
            raise ValueError("ReflexBench-TR records must have language='tr'")
        if record["origin"] not in VALID_ORIGINS:
            raise ValueError("origin must be native, translated, or synthetic")
        if record["task"] != "binary" or not isinstance(record["label"], bool):
            raise ValueError("negation records must be binary with boolean labels")
        if not all(isinstance(record[field], str) and record[field].strip() for field in ("state", "question", "pair_id")):
            raise ValueError("state, question, and pair_id must be non-empty strings")
        identifiers.add(identifier)
        pairs[record["pair_id"]].append(record)

    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"minimal pair {pair_id!r} must contain exactly two records")
        if pair[0]["question"] != pair[1]["question"]:
            raise ValueError(f"minimal pair {pair_id!r} must use the same question")
        if pair[0]["label"] == pair[1]["label"]:
            raise ValueError(f"minimal pair {pair_id!r} must flip its label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    records = load_records(args.path)
    validate(records)
    print(f"validated {len(records)} records and {len(records) // 2} minimal pairs")


if __name__ == "__main__":
    main()

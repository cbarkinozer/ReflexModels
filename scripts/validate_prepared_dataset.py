"""Check provenance, duplicate text, and split leakage in prepared JSONL data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED = {"id", "language", "origin", "source_dataset", "source_partition", "text", "label", "label_name"}
ORIGINS = {"native", "translated", "synthetic"}
SPLITS = {"train", "dev", "test"}


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(record, dict) or REQUIRED - record.keys():
                raise ValueError(f"{path}:{number}: missing required prepared-data fields")
            records.append(record)
    if not records:
        raise ValueError("no prepared records supplied")
    return records


def validate(records: list[dict[str, Any]], *, reject_text_duplicates: bool = False) -> int:
    ids: set[str] = set()
    text_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        identifier, language, origin = record["id"], record["language"], record["origin"]
        split, text = record["source_partition"], record["text"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("every prepared-data id must be unique")
        if language not in {"tr", "en"}:
            raise ValueError("language must be tr or en")
        if origin not in ORIGINS:
            raise ValueError("origin must be native, translated, or synthetic")
        if split not in SPLITS:
            raise ValueError("source_partition must be train, dev, or test")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(record["label"], int) or not isinstance(record["label_name"], str):
            raise ValueError("label must be an integer and label_name a string")
        ids.add(identifier)
        text_hash = hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()
        text_splits[text_hash].add(split)
        source_id = str(record.get("source_id", identifier))
        source_splits[(str(record["source_dataset"]), source_id)].add(split)
    duplicate_text_count = sum(len(splits) > 1 for splits in text_splits.values())
    if reject_text_duplicates and duplicate_text_count:
        raise ValueError("identical normalized text occurs in multiple splits")
    if any(len(splits) > 1 for splits in source_splits.values()):
        raise ValueError("a source example occurs in multiple splits")
    return duplicate_text_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--strict-text-duplicates", action="store_true")
    args = parser.parse_args()
    records = read_records(args.paths)
    duplicate_count = validate(records, reject_text_duplicates=args.strict_text_duplicates)
    print(f"validated {len(records)} prepared records; {duplicate_count} cross-split normalized-text duplicates flagged")


if __name__ == "__main__":
    main()

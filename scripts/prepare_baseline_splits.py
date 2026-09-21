"""Create a run-local, auditable overlap-sensitivity copy of prepared splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reflexmodels.split_policy import apply_overlap_policy


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"{path}:{number}: record needs text")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: contains no records")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--policy", choices=["report_only", "drop_train_dev_overlapping_test"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    supplied = {name: read_jsonl(getattr(args, name)) for name in ("train", "dev", "test")}
    output, removed = apply_overlap_policy(supplied, policy=args.policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in output.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", records)
    manifest = {
        "policy": args.policy,
        "input_counts": {name: len(records) for name, records in supplied.items()},
        "output_counts": {name: len(records) for name, records in output.items()},
        "removed_counts": removed,
        "test_split_copied_without_filtering": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

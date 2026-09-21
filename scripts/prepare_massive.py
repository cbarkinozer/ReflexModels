"""Prepare aligned MASSIVE Turkish/English intent files for fixed-label baselines.

Download MASSIVE from its official source separately. Pass its ``tr-TR.jsonl``
and ``en-US.jsonl`` files; raw source data is intentionally ignored by Git.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

LOCALES = {"tr": "tr-TR", "en": "en-US"}
SPLITS = {"train", "dev", "test"}


def load_locale(path: Path, expected_locale: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            identifier, locale = raw["id"], raw["locale"]
            partition, intent, utterance = raw["partition"], raw["intent"], raw["utt"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{number}: invalid MASSIVE record") from error
        if locale != expected_locale:
            raise ValueError(f"{path}:{number}: expected locale {expected_locale}, got {locale}")
        if partition not in SPLITS or not all(isinstance(value, str) and value for value in (identifier, intent, utterance)):
            raise ValueError(f"{path}:{number}: invalid id, intent, utterance, or partition")
        records.append({"id": identifier, "partition": partition, "intent": intent, "utt": utterance})
    if not records:
        raise ValueError(f"{path}: contains no MASSIVE records")
    return records


def prepare(locale_records: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Return language/split keys mapped to normalized records.

    Intent IDs are derived from the intersection of labels that exist in both
    locales. This avoids a silent mismatch if locale data are mixed by version.
    """
    intents = set(record["intent"] for record in locale_records["tr"])
    intents &= set(record["intent"] for record in locale_records["en"])
    if not intents:
        raise ValueError("Turkish and English MASSIVE files have no shared intents")
    label_names = sorted(intents)
    label_ids = {name: index for index, name in enumerate(label_names)}
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for language, records in locale_records.items():
        for record in records:
            if record["intent"] not in label_ids:
                continue
            result[f"{language}_{record['partition']}"].append({
                "id": f"massive-{LOCALES[language]}-{record['id']}",
                "language": language,
                "origin": "translated",
                "source_dataset": "MASSIVE",
                "source_id": record["id"],
                "source_locale": LOCALES[language],
                "source_partition": record["partition"],
                "text": record["utt"],
                "label": label_ids[record["intent"]],
                "label_name": record["intent"],
            })
    result["labels"] = [{"id": index, "name": name} for name, index in label_ids.items()]
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tr", type=Path, required=True, help="MASSIVE tr-TR.jsonl")
    parser.add_argument("--en", type=Path, required=True, help="MASSIVE en-US.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepared = prepare({"tr": load_locale(args.tr, LOCALES["tr"]), "en": load_locale(args.en, LOCALES["en"])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, records in prepared.items():
        path = args.output_dir / ("labels.json" if key == "labels" else f"massive_{key}.jsonl")
        if key == "labels":
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
            )


if __name__ == "__main__":
    main()

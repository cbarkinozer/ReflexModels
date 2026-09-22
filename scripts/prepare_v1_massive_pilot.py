"""Build a small Turkish V1 pilot from MASSIVE train/dev intent records.

The generated options and insufficient-option examples are synthetic. This
pilot tests the V1 training path; it is not native Turkish benchmark data or a
claim of unseen-label generalization. MASSIVE test is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reflexmodels.split_policy import normalized_text_hash


QUESTION = "Kullanıcının isteğine en uygun işlem hangisi?"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_texts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first record for each normalized utterance within a split."""
    seen: set[str] = set()
    result = []
    for record in records:
        digest = normalized_text_hash(record["text"])
        if digest not in seen:
            result.append(record)
            seen.add(digest)
    return result


def balanced_sample(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin sample by intent label, with stable per-label shuffling."""
    if limit < 1:
        raise ValueError("source-record limit must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["label_name"]].append(record)
    rng = random.Random(seed)
    labels = sorted(groups)
    rng.shuffle(labels)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(limit, len(records)):
        advanced = False
        for label in labels:
            if groups[label]:
                selected.append(groups[label].pop())
                advanced = True
                if len(selected) >= limit:
                    break
        if not advanced:
            break
    return selected


def option_record(source: dict[str, Any], descriptions: dict[str, str], *, split: str, answerable: bool, seed: int) -> dict[str, Any]:
    correct = source["label_name"]
    if correct not in descriptions:
        raise ValueError(f"missing Turkish description for intent {correct!r}")
    digest = hashlib.sha256(f"{seed}:{source['id']}:{answerable}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    distractors = [name for name in descriptions if name != correct]
    same_domain = [name for name in distractors if name.split("_")[0] == correct.split("_")[0]]
    rng.shuffle(same_domain)
    rng.shuffle(distractors)
    selected = [correct] if answerable else []
    for name in same_domain + distractors:
        if name not in selected:
            selected.append(name)
        if len(selected) == 4:
            break
    rng.shuffle(selected)
    suffix = "answerable" if answerable else "insufficient"
    return {
        "id": f"v1-{source['id']}-{suffix}", "language": "tr", "origin": "synthetic",
        "source": "MASSIVE 1.0 tr-TR", "source_id": source["source_id"],
        "source_origin": source["origin"], "split": split,
        "state": source["text"], "question": QUESTION,
        "options": {name: descriptions[name] for name in selected},
        "answerable": answerable, "correct_option": correct if answerable else None,
    }


def build_records(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], descriptions: dict[str, str],
    *, train_limit: int, validation_limit: int, seed: int,
) -> list[dict[str, Any]]:
    # Keep every validation utterance disjoint from the selected training
    # utterances by normalized text. No test or frozen benchmark is consulted.
    selected_train = balanced_sample(unique_texts(train), train_limit, seed)
    train_hashes = {normalized_text_hash(record["text"]) for record in selected_train}
    eligible_validation = [record for record in validation if normalized_text_hash(record["text"]) not in train_hashes]
    selected_validation = balanced_sample(unique_texts(eligible_validation), validation_limit, seed + 1)
    if not selected_validation:
        raise ValueError("no validation records remain after train/validation deduplication")
    result = []
    for split, sources in (("train", selected_train), ("validation", selected_validation)):
        for source in sources:
            result.append(option_record(source, descriptions, split=split, answerable=True, seed=seed))
            result.append(option_record(source, descriptions, split=split, answerable=False, seed=seed))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "data" / "prepared" / "massive")
    parser.add_argument("--descriptions", type=Path, default=ROOT / "configs" / "massive_intent_descriptions_tr.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=600)
    parser.add_argument("--validation-limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    labels = {item["name"] for item in json.loads((args.prepared_dir / "labels.json").read_text(encoding="utf-8"))}
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    if set(descriptions) != labels:
        raise ValueError("Turkish descriptions must cover exactly the prepared MASSIVE intent labels")
    records = build_records(
        read_jsonl(args.prepared_dir / "massive_tr_train.jsonl"),
        read_jsonl(args.prepared_dir / "massive_tr_dev.jsonl"),
        descriptions, train_limit=args.train_limit, validation_limit=args.validation_limit, seed=args.seed,
    )
    from validate_v1_decision_data import validate
    validate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    print(f"wrote {len(records)} V1 synthetic pilot records to {args.output}")


if __name__ == "__main__":
    main()

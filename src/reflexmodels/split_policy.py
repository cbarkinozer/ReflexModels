"""Predeclared, auditable split policies for fixed-label baselines."""

from __future__ import annotations

import hashlib
from typing import Any


def normalized_text_hash(text: str) -> str:
    """Return the normalization used solely for duplicate-overlap reporting."""
    return hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()


def apply_overlap_policy(
    splits: dict[str, list[dict[str, Any]]], *, policy: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Apply a documented policy without ever editing the supplied test split.

    ``report_only`` retains official data. ``drop_train_dev_overlapping_test``
    removes train/dev records whose normalized text is present in test. It is a
    sensitivity dataset, not a replacement for official MASSIVE evaluation.
    """
    required = {"train", "dev", "test"}
    if set(splits) != required:
        raise ValueError("splits must contain exactly train, dev, and test")
    if policy not in {"report_only", "drop_train_dev_overlapping_test"}:
        raise ValueError("unsupported overlap policy")
    test_hashes = {normalized_text_hash(record["text"]) for record in splits["test"]}
    filtered: dict[str, list[dict[str, Any]]] = {"test": list(splits["test"])}
    removed = {"train": 0, "dev": 0, "test": 0}
    for split in ("train", "dev"):
        records = list(splits[split])
        if policy == "drop_train_dev_overlapping_test":
            retained = [record for record in records if normalized_text_hash(record["text"]) not in test_hashes]
            removed[split] = len(records) - len(retained)
            filtered[split] = retained
        else:
            filtered[split] = records
    return filtered, removed

"""Inter-annotator agreement and frozen-record ingestion for ReflexBench-TR.

Reads the same annotation TSV that ``lint_annotation_template.py`` validates,
computes raw agreement and Cohen's kappa per ``task_family`` over rows where
both independent annotators have recorded a label (per ANNOTATION.md's
two-annotator-plus-adjudication protocol), and can ingest only the rows that
have reached a final ``adjudicated_label`` into a frozen JSONL benchmark
file. Rows that are still blank or only drafted are left untouched; this
script never invents or edits annotation content, and never overwrites an
existing frozen file.

Every family is ingested into the same ``options`` / ``correct_option`` /
``answerable`` contract as ``V1_DATA.md``, with ``split: "test"``, so the
frozen file is directly consumable by ``scripts/validate_v1_decision_data.py``
and ``scripts/evaluate_v1_decision_model.py`` without any conversion step --
including ``answerability`` items, which carry real runtime options the same
way the V1 answerability head actually sees them at inference. An
``answerability`` item adjudicated as ``"insufficient"``
(``lint_annotation_template.INSUFFICIENT_LABEL``) is emitted with
``answerable: false, correct_option: null``; every other item is emitted
with ``answerable: true`` and ``correct_option`` set to the adjudicated
option key.

Every record also carries a ``provenance`` block (annotator/adjudicator ids,
dates, guideline version, and the row's ``notes`` as adjudication rationale)
so the frozen benchmark stays auditable even though annotator blindness
cannot be mechanically enforced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


LINT = _load_module("lint_annotation_template", ROOT / "scripts" / "lint_annotation_template.py")


def cohens_kappa(annotation_1: list[str], annotation_2: list[str]) -> float | None:
    """Return two-rater Cohen's kappa, or None when it is mathematically undefined.

    Kappa is undefined when the two raters' combined labels only ever use one
    category: chance agreement is then 1.0 and (observed - chance) / (1 -
    chance) is 0/0. That is a non-informative batch, not a perfect score, so
    callers must not read ``None`` as ``1.0``.
    """
    if not annotation_1 or len(annotation_1) != len(annotation_2):
        raise ValueError("annotation_1 and annotation_2 must be non-empty and equal length")
    n = len(annotation_1)
    categories = sorted(set(annotation_1) | set(annotation_2))
    observed_agreement = sum(a == b for a, b in zip(annotation_1, annotation_2)) / n
    expected_agreement = sum(
        (annotation_1.count(category) / n) * (annotation_2.count(category) / n) for category in categories
    )
    if expected_agreement >= 1.0:
        return None
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def agreement_by_family(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Return per-``task_family`` agreement stats over double-annotated-or-later rows."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        stage = LINT.row_stage(row)
        if stage in ("double_annotated", "adjudicated"):
            grouped[row["task_family"]].append(row)
    stats: dict[str, dict[str, Any]] = {}
    for family, family_rows in grouped.items():
        annotation_1 = [row["annotation_1"] for row in family_rows]
        annotation_2 = [row["annotation_2"] for row in family_rows]
        agreed = sum(a == b for a, b in zip(annotation_1, annotation_2))
        kappa = cohens_kappa(annotation_1, annotation_2)
        stats[family] = {
            "n": len(family_rows),
            "raw_agreement": agreed / len(family_rows),
            "cohens_kappa": kappa,
            "kappa_defined": kappa is not None,
        }
    return stats


def select_adjudicated(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if LINT.row_stage(row) == "adjudicated"]


def _provenance(row: dict[str, str]) -> dict[str, str | None]:
    return {
        "annotator_1_id": row["annotator_1_id"],
        "annotator_2_id": row["annotator_2_id"],
        "adjudicator_id": row["adjudicator_id"],
        "annotation_date_1": row["annotation_date_1"],
        "annotation_date_2": row["annotation_date_2"],
        "adjudication_date": row["adjudication_date"],
        "guideline_version": row["guideline_version"],
        "notes": row["notes"] or None,
    }


def to_frozen_records(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert adjudicated TSV rows into the executable V1_DATA.md benchmark schema."""
    records = []
    for row in rows:
        family = row["task_family"]
        label = row["adjudicated_label"]
        options = LINT.parse_options(row["options_json"], item_id=row["item_id"], task_family=family)
        is_insufficient = family == LINT.ANSWERABILITY_FAMILY and label == LINT.INSUFFICIENT_LABEL
        records.append({
            "id": row["item_id"], "language": "tr", "origin": "native",
            "task_family": family, "state": row["state"], "question": row["question"],
            "source": f"reflexbench-tr-{family}", "split": "test",
            "options": options,
            "answerable": not is_insufficient,
            "correct_option": None if is_insufficient else label,
            "provenance": _provenance(row),
        })
    return records


def write_frozen(records: list[dict[str, Any]], path: Path) -> None:
    """Write adjudicated records as JSONL. Refuses to overwrite an existing frozen file.

    Uses an exclusive-create open ("x" mode) rather than an exists-check-then-write,
    so the never-overwrite guarantee holds even if two processes race on the same path.
    """
    if not records:
        raise ValueError("no adjudicated records to ingest; nothing was written")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"{path} already exists; frozen benchmark files are never overwritten. "
            "Choose a new path (or remove the old one deliberately, outside this tool) if this is intentional."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--ingest", type=Path, help="Write adjudicated rows as frozen JSONL to this new path")
    args = parser.parse_args()
    rows = LINT.load_rows(args.path)
    LINT.validate(rows)

    stats = agreement_by_family(rows)
    if not stats:
        print("no double-annotated or adjudicated rows yet")
    for family in sorted(stats):
        values = stats[family]
        kappa_text = f"{values['cohens_kappa']:.3f}" if values["kappa_defined"] else "undefined (single category)"
        print(f"{family}: n={values['n']} raw_agreement={values['raw_agreement']:.3f} cohens_kappa={kappa_text}")

    adjudicated = select_adjudicated(rows)
    print(f"{len(adjudicated)} of {len(rows)} rows are adjudicated")
    if args.ingest:
        write_frozen(to_frozen_records(adjudicated), args.ingest)
        print(f"wrote {len(adjudicated)} frozen records to {args.ingest}")


if __name__ == "__main__":
    main()

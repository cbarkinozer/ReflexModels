"""Machine-validate the native-Turkish ReflexBench-TR annotation TSV.

The template in ``benchmarks/reflexbench_tr/annotation_templates/items.tsv``
starts blank (per its README, item text is authored independently by native
speakers, never translated or model-generated) and fills in over time:
blank -> drafted -> double_annotated -> adjudicated. This linter enforces
that every row is in one of those valid stages and never a partial or
inconsistent one, without requiring the whole file to be finished.

Every task family carries structured ``options_json`` (a JSON object of at
least two ``{"option_id": "option text"}`` entries): this is required so the
ingested benchmark can exercise the real V1 decision path -- state +
question + supplied options -- the same way the V1 answerability head and
``evaluate_v1_decision_model.py`` actually consume it, not just a free-text
label a model was never asked to choose between.

- Most families: every annotation and the final ``adjudicated_label`` must
  name one of the supplied option keys (a correct option).
- ``answerability``: a label may also be the reserved value
  ``"insufficient"``, meaning none of the supplied options is answerable
  from the given state (``answerable: false, correct_option: null`` in the
  frozen record) -- rather than one specific option being correct. An
  ``answerability`` item's own ``options_json`` must not itself define an
  option literally named ``"insufficient"``, to keep that meaning
  unambiguous.

Once two annotators have recorded a label (``annotator_1_id``,
``annotator_2_id``, and their ISO-8601 dates are then required), and once an
adjudicator sets a final label (``adjudicator_id`` and its ISO-8601 date are
then required), those audit fields make the process auditable even though
annotator blindness cannot be mechanically enforced. Audit fields may never
appear on a row whose corresponding annotation/adjudication content is
absent ("dangling" metadata).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = [
    "item_id", "task_family", "state", "question", "options_json",
    "annotation_1", "annotation_2", "adjudicated_label", "notes",
    "annotator_1_id", "annotator_2_id", "adjudicator_id",
    "annotation_date_1", "annotation_date_2", "adjudication_date",
    "guideline_version",
]
CHOICE_FAMILIES = {"intent", "tool_routing", "unseen_option", "code_switching", "relevance"}
ANSWERABILITY_FAMILY = "answerability"
VALID_TASK_FAMILIES = CHOICE_FAMILIES | {ANSWERABILITY_FAMILY}
VALID_STAGES = {"blank", "drafted", "double_annotated", "adjudicated"}
INSUFFICIENT_LABEL = "insufficient"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f"{path}: header must be exactly {REQUIRED_COLUMNS}, found {reader.fieldnames}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{path}: contains no rows")
    return rows


def parse_options(raw: str, *, item_id: str, task_family: str = "") -> dict[str, str]:
    try:
        options = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{item_id}: options_json must be valid JSON") from error
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError(f"{item_id}: options_json must be an object with at least two options")
    for key, value in options.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError(f"{item_id}: option keys and text must be non-empty strings")
    if task_family == ANSWERABILITY_FAMILY and INSUFFICIENT_LABEL in options:
        raise ValueError(f"{item_id}: answerability options_json must not define an option named {INSUFFICIENT_LABEL!r}")
    return options


def _label_valid(label: str, *, task_family: str, options: dict[str, str] | None) -> bool:
    if options is None:
        return False
    if task_family == ANSWERABILITY_FAMILY and label == INSUFFICIENT_LABEL:
        return True
    return label in options


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def row_stage(row: dict[str, str]) -> str:
    """Classify a row's authoring stage, or 'invalid' if fields are inconsistent."""
    family = row["task_family"]
    if family not in VALID_TASK_FAMILIES:
        return "invalid"
    state, question, options_raw = row["state"], row["question"], row["options_json"]
    annotation_1, annotation_2, adjudicated = row["annotation_1"], row["annotation_2"], row["adjudicated_label"]
    has_annotation_1, has_annotation_2, has_adjudication = bool(annotation_1), bool(annotation_2), bool(adjudicated)
    annotator_1_id, annotator_2_id = row["annotator_1_id"], row["annotator_2_id"]
    date_1, date_2, guideline_version = row["annotation_date_1"], row["annotation_date_2"], row["guideline_version"]
    adjudicator_id, adjudication_date = row["adjudicator_id"], row["adjudication_date"]
    annotation_1_audit = bool(annotator_1_id or date_1)
    annotation_2_audit = bool(annotator_2_id or date_2)
    adjudication_audit = bool(adjudicator_id or adjudication_date)

    has_any_content = bool(
        state or question or options_raw or has_annotation_1 or has_annotation_2 or has_adjudication
        or annotation_1_audit or annotation_2_audit or adjudication_audit or guideline_version
    )
    if not has_any_content:
        return "blank"
    if not state or not question or not options_raw:
        return "invalid"
    try:
        options = parse_options(options_raw, item_id=row["item_id"], task_family=family)
    except ValueError:
        return "invalid"

    if has_annotation_1 != has_annotation_2:
        return "invalid"
    if has_adjudication and not (has_annotation_1 and has_annotation_2):
        return "invalid"
    # No audit metadata may dangle on a row whose corresponding content is absent.
    if annotation_1_audit and not has_annotation_1:
        return "invalid"
    if annotation_2_audit and not has_annotation_2:
        return "invalid"
    if adjudication_audit and not has_adjudication:
        return "invalid"
    if guideline_version and not (has_annotation_1 and has_annotation_2):
        return "invalid"

    if has_annotation_1:
        if not _label_valid(annotation_1, task_family=family, options=options):
            return "invalid"
        if not _label_valid(annotation_2, task_family=family, options=options):
            return "invalid"
        if not (annotator_1_id and annotator_2_id and date_1 and date_2 and guideline_version):
            return "invalid"
        if not (_is_iso_date(date_1) and _is_iso_date(date_2)):
            return "invalid"

    if has_adjudication:
        if not _label_valid(adjudicated, task_family=family, options=options):
            return "invalid"
        if not (adjudicator_id and adjudication_date):
            return "invalid"
        if not _is_iso_date(adjudication_date):
            return "invalid"
        return "adjudicated"

    if has_annotation_1 and has_annotation_2:
        return "double_annotated"

    return "drafted"


def validate(rows: list[dict[str, str]]) -> None:
    identifiers: set[str] = set()
    for row in rows:
        if not row["item_id"]:
            raise ValueError("every row must have a non-empty item_id")
        if row["item_id"] in identifiers:
            raise ValueError(f"duplicate item_id: {row['item_id']}")
        identifiers.add(row["item_id"])
        if row["task_family"] not in VALID_TASK_FAMILIES:
            raise ValueError(f"{row['item_id']}: task_family must be one of {sorted(VALID_TASK_FAMILIES)}")
        if row_stage(row) == "invalid":
            raise ValueError(
                f"{row['item_id']}: inconsistent {row['task_family']} row -- state/question/options_json, "
                "annotation_1/annotation_2 (with annotator ids, ISO-8601 dates, and guideline_version), and "
                "adjudicated_label (with adjudicator id and ISO-8601 date) must each be filled together, in "
                "that order, with no dangling audit fields on an unfilled step; every label must name a "
                "supplied option key (or 'insufficient' for an unanswerable answerability item)"
            )


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Return per-task-family stage counts. Assumes ``validate`` already passed."""
    by_family: dict[str, Counter] = {}
    for row in rows:
        counter = by_family.setdefault(row["task_family"], Counter())
        counter[row_stage(row)] += 1
    return {
        family: {stage: counter.get(stage, 0) for stage in ("blank", "drafted", "double_annotated", "adjudicated")}
        for family, counter in by_family.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.path)
    validate(rows)
    summary = summarize(rows)
    print(f"validated {len(rows)} rows across {len(summary)} task families")
    for family in sorted(summary):
        counts = summary[family]
        print(f"  {family}: blank={counts['blank']} drafted={counts['drafted']} "
              f"double_annotated={counts['double_annotated']} adjudicated={counts['adjudicated']}")


if __name__ == "__main__":
    main()

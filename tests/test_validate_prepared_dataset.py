import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_prepared_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_prepared_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def record(identifier, split, text, *, source_id=None):
    return {
        "id": identifier, "language": "tr", "origin": "translated", "source_dataset": "MASSIVE",
        "source_partition": split, "text": text, "label": 0, "label_name": "alarm_set",
        **({"source_id": source_id} if source_id else {}),
    }


class PreparedDatasetValidationTests(unittest.TestCase):
    def test_accepts_distinct_text_and_source_records(self):
        MODULE.validate([record("a", "train", "Alarm kur", source_id="a"), record("b", "test", "Saat kaç?", source_id="b")])

    def test_rejects_identical_text_across_splits(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            MODULE.validate(
                [record("a", "train", "Alarm kur"), record("b", "test", " alarm KUR ")],
                reject_text_duplicates=True,
            )

    def test_reports_but_does_not_reject_text_duplicates_by_default(self):
        count = MODULE.validate([record("a", "train", "Alarm kur"), record("b", "test", " alarm KUR ")])
        self.assertEqual(count, 1)

    def test_rejects_source_id_leakage(self):
        with self.assertRaisesRegex(ValueError, "source example"):
            MODULE.validate([record("a", "train", "Alarm kur", source_id="x"), record("b", "test", "Saat kaç?", source_id="x")])

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_v1_decision_data.py"
SPEC = importlib.util.spec_from_file_location("validate_v1_decision_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def record(identifier="tr-1", **overrides):
    value = {
        "id": identifier, "language": "tr", "origin": "native", "source": "curated-v1",
        "split": "train", "state": "Durum.", "question": "Ne yapılmalı?",
        "options": {"a": "A seçeneği", "b": "B seçeneği"},
        "answerable": True, "correct_option": "a",
    }
    value.update(overrides)
    return value


class V1DecisionDataValidationTests(unittest.TestCase):
    def test_accepts_answerable_and_unanswerable_records(self):
        MODULE.validate([record(), record("tr-2", answerable=False, correct_option=None)])

    def test_rejects_correct_option_outside_options(self):
        with self.assertRaisesRegex(ValueError, "correct_option"):
            MODULE.validate([record(correct_option="missing")])

    def test_rejects_label_for_unanswerable_record(self):
        with self.assertRaisesRegex(ValueError, "null"):
            MODULE.validate([record(answerable=False, correct_option="a")])

    def test_rejects_frozen_test_split(self):
        with self.assertRaisesRegex(ValueError, "split"):
            MODULE.validate([record(split="test")])


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_reflexbench_tr.py"
SPEC = importlib.util.spec_from_file_location("validate_reflexbench_tr", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def record(identifier, label, *, pair_id="pair", question="Doğru mu?"):
    return {
        "id": identifier, "language": "tr", "origin": "native", "task": "binary",
        "state": "Durum.", "question": question, "label": label, "pair_id": pair_id,
    }


class ReflexBenchValidationTests(unittest.TestCase):
    def test_valid_pair(self):
        MODULE.validate([record("a", True), record("b", False)])

    def test_rejects_pair_without_label_flip(self):
        with self.assertRaisesRegex(ValueError, "flip"):
            MODULE.validate([record("a", True), record("b", True)])

    def test_rejects_missing_provenance(self):
        invalid = record("a", True)
        del invalid["origin"]
        with self.assertRaisesRegex(ValueError, "origin"):
            MODULE.validate([invalid, record("b", False)])


if __name__ == "__main__":
    unittest.main()

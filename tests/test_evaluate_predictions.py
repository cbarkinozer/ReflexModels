import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_predictions.py"
SPEC = importlib.util.spec_from_file_location("evaluate_predictions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class PredictionLoadingTests(unittest.TestCase):
    def test_groups_predictions_by_language(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"id": "tr-1", "language": "tr", "label": 0, "probabilities": [0.8, 0.2]}),
                    json.dumps({"id": "en-1", "language": "en", "label": 1, "probabilities": [0.1, 0.9]}),
                ]),
                encoding="utf-8",
            )
            grouped = MODULE.load_predictions(path)
        self.assertEqual(grouped["tr"], ([0], [[0.8, 0.2]]))
        self.assertEqual(grouped["en"], ([1], [[0.1, 0.9]]))

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            row = {"id": "same", "language": "tr", "label": 0, "probabilities": [0.8, 0.2]}
            path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_predictions(path)


if __name__ == "__main__":
    unittest.main()

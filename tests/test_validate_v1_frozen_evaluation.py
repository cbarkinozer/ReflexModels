import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_v1_frozen_evaluation.py"
SPEC = importlib.util.spec_from_file_location("validate_v1_frozen_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

EVALUATOR_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_v1_decision_model.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location("evaluate_v1_for_validation_tests", EVALUATOR_SCRIPT)
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
assert EVALUATOR_SPEC and EVALUATOR_SPEC.loader
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


class ValidateV1FrozenEvaluationTests(unittest.TestCase):
    def write_evaluation(self, directory: Path):
        records = [
            {"id": "native-tool", "split": "test", "language": "tr", "origin": "native", "task_family": "tool_routing",
             "options": {"a": "A", "b": "B"}, "answerable": True, "correct_option": "a"},
            {"id": "synthetic-answer", "split": "test", "language": "tr", "origin": "synthetic", "task_family": "answerability",
             "options": {"a": "A", "b": "B"}, "answerable": False, "correct_option": None},
        ]
        metrics, predictions = EVALUATOR.evaluate_predictions(
            records, [{"a": 2.0, "b": 0.0}, {"a": 0.0, "b": 2.0}], [0.9, 0.1], temperature=1.0, threshold=0.5,
        )
        metrics["model_dir"] = "model"
        metrics["test_data"] = "test.jsonl"
        (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (directory / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )

    def test_accepts_metrics_recomputed_from_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_evaluation(directory)
            self.assertEqual(MODULE.validate(directory), {"predictions": 2, "languages": 1, "origins": 2})

    def test_rejects_tampered_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_evaluation(directory)
            metrics_path = directory / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["by_origin"]["native"]["decision"]["accuracy"] = 0.0
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.validate(directory)

    def test_rejects_duplicate_prediction_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_evaluation(directory)
            predictions_path = directory / "predictions.jsonl"
            first = predictions_path.read_text(encoding="utf-8").splitlines()[0]
            predictions_path.write_text(first + "\n" + first + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                MODULE.validate(directory)

    def test_rejects_invalid_probability_or_fallback_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_evaluation(directory)
            predictions_path = directory / "predictions.jsonl"
            rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["option_probabilities"] = {"a": 0.8, "b": 0.8}
            predictions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sum to one"):
                MODULE.validate(directory)
            rows[0]["option_probabilities"] = {"a": 0.8, "b": 0.2}
            rows[0]["accepted"] = False
            predictions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted"):
                MODULE.validate(directory)


if __name__ == "__main__":
    unittest.main()

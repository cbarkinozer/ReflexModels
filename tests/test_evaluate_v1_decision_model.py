import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_v1_decision_model.py"
SPEC = importlib.util.spec_from_file_location("evaluate_v1_decision_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

from reflexmodels.split_policy import normalized_text_hash


def record(identifier, language, answerable, correct_option):
    return {
        "id": identifier, "split": "test", "language": language, "origin": "native",
        "options": {"a": "Evet", "b": "Hayır"}, "answerable": answerable,
        "correct_option": correct_option,
    }


class V1FrozenEvaluationTests(unittest.TestCase):
    def test_fixed_calibration_and_separate_language_metrics(self):
        records = [
            record("tr-1", "tr", True, "a"), record("tr-2", "tr", False, None),
            record("en-1", "en", True, "b"),
        ]
        scores = [{"a": 3.0, "b": 1.0}, {"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 4.0}]
        metrics, predictions = MODULE.evaluate_predictions(
            records, scores, [0.9, 0.1, 0.8], temperature=2.0, threshold=0.5,
        )
        self.assertEqual(set(metrics["by_language"]), {"tr", "en"})
        self.assertEqual(set(metrics["by_origin"]), {"native"})
        self.assertEqual(metrics["calibration_temperature"], 2.0)
        self.assertEqual(metrics["answerability_threshold"], 0.5)
        self.assertEqual(metrics["by_language"]["tr"]["coverage"]["coverage"], 0.5)
        self.assertEqual(metrics["by_language"]["en"]["decision"]["accuracy"], 1.0)
        self.assertFalse(predictions[1]["accepted"])
        self.assertAlmostEqual(predictions[0]["option_probabilities"]["a"], 0.73105858, places=6)

    def test_reports_task_family_metrics_when_frozen_benchmark_supplies_them(self):
        records = [
            {**record("tr-tool", "tr", True, "a"), "task_family": "tool_routing"},
            {**record("tr-answer", "tr", False, None), "task_family": "answerability"},
        ]
        metrics, _ = MODULE.evaluate_predictions(
            records, [{"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 2.0}], [0.9, 0.1], temperature=1, threshold=0.5,
        )
        self.assertEqual(set(metrics["by_task_family"]), {"tool_routing", "answerability"})
        self.assertEqual(metrics["by_task_family"]["tool_routing"]["decision"]["accuracy"], 1.0)
        self.assertIsNone(metrics["by_task_family"]["answerability"]["decision"])

    def test_reports_native_and_synthetic_origins_separately(self):
        native = record("tr-native", "tr", True, "a")
        synthetic = {**record("tr-synthetic", "tr", False, None), "origin": "synthetic"}
        metrics, _ = MODULE.evaluate_predictions(
            [native, synthetic], [{"a": 3.0, "b": 1.0}, {"a": 1.0, "b": 3.0}], [0.9, 0.1],
            temperature=1, threshold=0.5,
        )
        self.assertEqual(set(metrics["by_origin"]), {"native", "synthetic"})
        self.assertEqual(metrics["by_origin"]["native"]["record_count"], 1)
        self.assertEqual(metrics["by_origin"]["synthetic"]["record_count"], 1)

    def test_rejects_partially_labeled_task_family(self):
        records = [record("tr-1", "tr", True, "a"), {**record("tr-2", "tr", False, None), "task_family": "answerability"}]
        with self.assertRaisesRegex(ValueError, "task_family"):
            MODULE.evaluate_predictions(
                records, [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], [0.9, 0.1], temperature=1, threshold=0.5,
            )

    def test_rejects_validation_records(self):
        bad = record("r", "tr", True, "a")
        bad["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "split=test"):
            MODULE.evaluate_predictions([bad], [{"a": 1.0, "b": 0.0}], [0.9], temperature=1, threshold=0.5)

    def test_rejects_option_mismatch_and_invalid_probability(self):
        one = record("r", "tr", True, "a")
        with self.assertRaisesRegex(ValueError, "scores do not match"):
            MODULE.evaluate_predictions([one], [{"a": 1.0}], [0.9], temperature=1, threshold=0.5)
        with self.assertRaisesRegex(ValueError, "invalid answerability"):
            MODULE.evaluate_predictions([one], [{"a": 1.0, "b": 0.0}], [float("nan")], temperature=1, threshold=0.5)

    def test_rejects_train_or_validation_state_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "fingerprints.json"
            manifest.write_text(json.dumps({
                "normalization": "strip+casefold+sha256 of state text",
                "train_state_hashes": [normalized_text_hash("Örnek durum")],
                "validation_state_hashes": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                MODULE.reject_training_overlap([{"id": "test-1", "state": " örnek durum "}], manifest)
            MODULE.reject_training_overlap([{"id": "test-2", "state": "Başka durum"}], manifest)

    def test_reproducibility_metadata_hashes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "backbone").mkdir()
            files = {
                root / "train_metadata.json": "metadata",
                root / "decision_heads.pt": "heads",
                root / "backbone" / "config.json": "config",
                root / "backbone" / "tokenizer_config.json": "tokenizer",
                root / "test.jsonl": "test data",
            }
            for path, content in files.items():
                path.write_text(content, encoding="utf-8")
            value = MODULE.reproducibility_metadata(root, root / "test.jsonl", torch_version="test")
            self.assertEqual(value["torch"], "test")
            self.assertEqual(value["test_data_sha256"], MODULE.file_sha256(root / "test.jsonl"))
            (root / "decision_heads.pt").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                MODULE.reproducibility_metadata(root, root / "test.jsonl", torch_version="test")


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_fixed_label_baseline.py"
SPEC = importlib.util.spec_from_file_location("fixed_label_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(runner)


class FixedLabelRunnerTests(unittest.TestCase):
    def test_select_run_and_label_count(self):
        config = {"runs": [{"name": "tr"}]}
        self.assertEqual(runner.select_run(config, "tr"), {"name": "tr"})
        splits = {name: [{"label": 0}, {"label": 1}] for name in ("train", "dev", "test")}
        self.assertEqual(runner.label_count(splits), 2)

    def test_uses_validation_split_for_development_data(self):
        run = {"train_split": "train.jsonl", "validation_split": "dev.jsonl", "test_split": "test.jsonl"}
        self.assertEqual(runner.split_filename(run, "dev"), "dev.jsonl")

    def test_rejects_non_contiguous_labels(self):
        splits = {name: [{"label": 1}] for name in ("train", "dev", "test")}
        with self.assertRaises(ValueError):
            runner.label_count(splits)

    def test_softmax_probabilities_sum_to_one(self):
        rows = runner.probabilities([[1.0, 2.0]])
        self.assertAlmostEqual(sum(rows[0]), 1.0)
        self.assertGreater(rows[0][1], rows[0][0])


if __name__ == "__main__":
    unittest.main()

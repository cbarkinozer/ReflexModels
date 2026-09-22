import importlib.util
import unittest
from pathlib import Path
import tempfile

import torch


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

    def test_calibration_preserves_all_prediction_rows(self):
        rows = [[0.8, 0.2], [0.1, 0.9]]
        scaled = runner.calibrated_probabilities(rows, 2.0)
        self.assertEqual(len(scaled), 2)
        self.assertTrue(all(len(row) == 2 for row in scaled))
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in scaled))

    def test_postprocessing_preflight_runs_before_training(self):
        runner.preflight_postprocessing()

    def test_epoch_checkpoint_survives_model_recreation(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch_1.pt"
            runner.save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, validation_nll=0.5, torch=torch)
            restored = torch.nn.Linear(2, 2)
            checkpoint = runner.load_checkpoint(path, model=restored, torch=torch)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["validation_nll"], 0.5)
        self.assertTrue(all(torch.equal(expected[key], restored.state_dict()[key]) for key in expected))


if __name__ == "__main__":
    unittest.main()

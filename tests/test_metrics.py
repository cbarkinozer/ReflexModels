import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from reflexmodels.metrics import classification_metrics


class ClassificationMetricTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_calibration_losses(self):
        metrics = classification_metrics([0, 1], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(metrics, {
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "nll": 0.0,
            "brier": 0.0,
            "ece": 0.0,
        })

    def test_metrics_validate_probability_rows(self):
        with self.assertRaises(ValueError):
            classification_metrics([0], [[0.7, 0.7]])

    def test_macro_f1_includes_an_unpredicted_class(self):
        metrics = classification_metrics([0, 1], [[0.9, 0.1], [0.8, 0.2]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()

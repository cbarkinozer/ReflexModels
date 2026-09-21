import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from reflexmodels import answerability_metrics, select_answerability_threshold, selective_accuracy


class AnswerabilityTests(unittest.TestCase):
    def test_perfect_answerability_has_perfect_accuracy(self):
        result = answerability_metrics([1, 0], [0.9, 0.1])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["answerable_recall"], 1.0)

    def test_selective_accuracy_reports_coverage(self):
        result = selective_accuracy([True, False, True], [0.9, 0.2, 0.8], threshold=0.75)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["selective_accuracy"], 1.0)

    def test_rejects_nonbinary_label(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            answerability_metrics([2], [0.5])

    def test_selects_validation_threshold_with_best_selective_accuracy(self):
        result = select_answerability_threshold(
            [True, False, True, False], [0.9, 0.8, 0.7, 0.1], min_coverage=0.25
        )
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["selective_accuracy"], 1.0)

    def test_rejects_impossible_coverage(self):
        with self.assertRaisesRegex(ValueError, "min_coverage"):
            select_answerability_threshold([True], [0.5], min_coverage=1.1)


if __name__ == "__main__":
    unittest.main()

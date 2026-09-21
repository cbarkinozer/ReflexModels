import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from reflexmodels.calibration import fit_temperature, reliability_bins, temperature_scale


class CalibrationTests(unittest.TestCase):
    def test_temperature_one_preserves_probabilities(self):
        values = [[0.8, 0.2], [0.3, 0.7]]
        self.assertEqual(temperature_scale(values, 1.0), values)

    def test_high_temperature_softens_distribution(self):
        scaled = temperature_scale([[0.9, 0.1]], 2.0)[0]
        self.assertLess(scaled[0], 0.9)
        self.assertGreater(scaled[1], 0.1)

    def test_fit_temperature_returns_positive_value(self):
        temperature = fit_temperature([0, 1], [[0.99, 0.01], [0.99, 0.01]])
        self.assertGreater(temperature, 1.0)

    def test_reliability_bins_report_only_populated_bins(self):
        bins = reliability_bins([0, 1], [[0.9, 0.1], [0.2, 0.8]], bins=2)
        self.assertEqual(bins[0]["count"], 2)
        self.assertEqual(bins[0]["accuracy"], 1.0)

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from reflexmodels.cpu_benchmark import measure_callable


class CpuBenchmarkTests(unittest.TestCase):
    def test_reports_positive_throughput_and_request_count(self):
        result = measure_callable(lambda: sum(range(10)), warmup=1, repetitions=3)
        self.assertEqual(result["requests"], 3)
        self.assertGreater(result["requests_per_second"], 0)
        self.assertGreaterEqual(result["python_peak_ram_bytes"], 0)

    def test_rejects_invalid_run_counts(self):
        with self.assertRaises(ValueError):
            measure_callable(lambda: None, repetitions=0)

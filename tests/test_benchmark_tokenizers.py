import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_tokenizers.py"
SPEC = importlib.util.spec_from_file_location("benchmark_tokenizers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class TokenizerMetricTests(unittest.TestCase):
    def test_summary_uses_per_example_rates_and_sequence_percentiles(self):
        result = MODULE.summarize(WhitespaceTokenizer(), ["bir iki", "üç dört beş altı"])
        self.assertEqual(result["examples"], 2)
        self.assertEqual(result["tokens_per_word"], 1.0)
        self.assertEqual(result["sequence_tokens_p50"], 3.0)
        self.assertAlmostEqual(result["sequence_tokens_p95"], 3.9)

    def test_percentile_rejects_empty_values(self):
        with self.assertRaises(ValueError):
            MODULE.percentile([], 0.5)

    def test_token_count_accepts_a_tokenizers_style_encoding(self):
        class Encoding:
            def __len__(self):
                return 3

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return Encoding()

        self.assertEqual(MODULE.token_count(Tokenizer(), "örnek"), 3)


if __name__ == "__main__":
    unittest.main()

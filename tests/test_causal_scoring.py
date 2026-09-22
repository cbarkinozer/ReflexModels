import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from reflexmodels import causal_scoring
from reflexmodels.causal_scoring import branch_prefix, candidate_sequences
from reflexmodels.types import Choice


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]


class CausalScoringTests(unittest.TestCase):
    def test_each_option_has_the_same_independent_prefix(self):
        sequences = candidate_sequences(
            FakeTokenizer(), state="Durum", question="Ne olmali?", options={"a": "Bir", "b": "Iki"}
        )
        prefix = [ord(character) for character in branch_prefix("Durum", "Ne olmali?")]
        self.assertEqual(sequences["a"][0][:len(prefix)], prefix)
        self.assertEqual(sequences["b"][0][:len(prefix)], prefix)
        self.assertEqual(sequences["a"][1], [ord(character) for character in "Bir"])
        self.assertEqual(sequences["b"][1], [ord(character) for character in "Iki"])

    def test_rejects_one_option(self):
        with self.assertRaisesRegex(ValueError, "two"):
            candidate_sequences(FakeTokenizer(), state="D", question="S", options={"a": "A"})

    def test_converts_v0_scores_to_a_typed_response(self):
        request = Choice(state="Durum", question="Hangisi?", options={"a": "Bir", "b": "Iki"})
        original = causal_scoring.option_logprobabilities
        causal_scoring.option_logprobabilities = lambda *args, **kwargs: {"a": 3.0, "b": 1.0}
        try:
            response = causal_scoring.score_causal_decision(
                request, request_id="r-1", model=object(), tokenizer=FakeTokenizer()
            )
        finally:
            causal_scoring.option_logprobabilities = original
        self.assertEqual(response["selected_option"], "a")
        self.assertAlmostEqual(sum(response["option_probabilities"].values()), 1.0)

    def test_tail_logit_optimization_matches_full_logits(self):
        class Full(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids):
                return SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids, 128).float() * 2)

        class Tail(Full):
            def forward(self, input_ids, logits_to_keep=0):
                output = super().forward(input_ids)
                return SimpleNamespace(logits=output.logits[:, -logits_to_keep:])

        kwargs = {"state": "A", "question": "B?", "options": {"a": "Yes", "b": "No"}}
        full = causal_scoring.option_logprobabilities(Full(), FakeTokenizer(), **kwargs)
        tail = causal_scoring.option_logprobabilities(Tail(), FakeTokenizer(), **kwargs)
        self.assertEqual(full, tail)


if __name__ == "__main__":
    unittest.main()

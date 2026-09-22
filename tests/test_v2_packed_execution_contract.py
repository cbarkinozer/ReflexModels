import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from reflexmodels import score_shared_state
from reflexmodels.packed_execution_contract import (
    EXACT_TOLERANCE, QUANTIZED_TOLERANCE, assert_packed_matches_oracle, reference_batches,
)


def deterministic_scorer(branch_text: str) -> float:
    """Any pure function of the branch text works: both sides see the same text."""
    return sum(ord(character) for character in branch_text) % 97


def balanced_scorer(branch_text: str) -> float:
    """Small, close-together scores so softmax output isn't saturated near 0/1."""
    return sum(ord(character) for character in branch_text) / 10000.0


class ReferenceBatchesTests(unittest.TestCase):
    def test_covers_binary_choice_and_ordinal(self):
        decision_types = {
            type(decision).__name__
            for batch in reference_batches()
            for decision in batch.decisions.values()
        }
        self.assertEqual(decision_types, {"Binary", "Choice", "Ordinal"})

    def test_includes_single_and_multi_decision_batches(self):
        sizes = {len(batch.decisions) for batch in reference_batches()}
        self.assertIn(1, sizes)
        self.assertTrue(any(size > 1 for size in sizes))

    def test_includes_option_order_permutation_of_an_existing_batch(self):
        option_sets = [
            tuple(decision.options.items())
            for batch in reference_batches()
            for decision in batch.decisions.values()
            if hasattr(decision, "options")
        ]
        forward = ("filesystem", "Yerel dosyaları oku"), ("web", "İnternette ara")
        backward = ("web", "İnternette ara"), ("filesystem", "Yerel dosyaları oku")
        self.assertIn(forward, option_sets)
        self.assertIn(backward, option_sets)

    def test_includes_duplicate_option_text_under_different_ids(self):
        texts_by_options = [
            list(decision.options.values())
            for batch in reference_batches()
            for decision in batch.decisions.values()
            if hasattr(decision, "options")
        ]
        self.assertTrue(any(len(values) != len(set(values)) for values in texts_by_options))

    def test_includes_turkish_orthography_and_long_state(self):
        states = [batch.state for batch in reference_batches()]
        self.assertTrue(any("İ" in state and "ı" in state for state in states))
        self.assertTrue(any(len(state) > 1000 for state in states))

    def test_includes_multi_decision_batch_with_varied_candidate_counts(self):
        candidate_counts = [
            sorted(
                len(decision.options) if hasattr(decision, "options")
                else len(decision.levels) if hasattr(decision, "levels")
                else 2  # Binary
                for decision in batch.decisions.values()
            )
            for batch in reference_batches()
            if len(batch.decisions) > 1
        ]
        self.assertTrue(any(len(set(counts)) == len(counts) and len(counts) >= 3 for counts in candidate_counts))


class AssertPackedMatchesOracleTests(unittest.TestCase):
    def test_passes_when_packed_executor_is_the_oracle_itself(self):
        # Stand-in for "no V2 implementation exists yet": the duplicated-branch
        # oracle trivially satisfies its own contract, which documents exactly
        # what a real packed executor must reproduce.
        for batch in reference_batches():
            assert_packed_matches_oracle(
                score_shared_state, batch, request_id="req-1", scorer=deterministic_scorer
            )

    def test_detects_wrong_selected_option(self):
        def broken(batch, *, request_id, scorer):
            result = score_shared_state(batch, request_id=request_id, scorer=scorer)
            first_id = next(iter(result))
            probabilities = result[first_id]["option_probabilities"]
            worst_option = min(probabilities, key=probabilities.__getitem__)
            result[first_id] = {**result[first_id], "selected_option": worst_option}
            return result

        with self.assertRaisesRegex(AssertionError, "selected_option"):
            assert_packed_matches_oracle(
                broken, reference_batches()[1], request_id="req-1", scorer=deterministic_scorer
            )

    def test_detects_missing_decision_id(self):
        def broken(batch, *, request_id, scorer):
            result = score_shared_state(batch, request_id=request_id, scorer=scorer)
            first_id = next(iter(result))
            del result[first_id]
            return result

        with self.assertRaisesRegex(AssertionError, "decision ids"):
            assert_packed_matches_oracle(
                broken, reference_batches()[1], request_id="req-1", scorer=deterministic_scorer
            )

    def test_detects_probability_drift_beyond_tolerance(self):
        def broken(batch, *, request_id, scorer):
            result = score_shared_state(batch, request_id=request_id, scorer=scorer)
            first_id = next(iter(result))
            probabilities = dict(result[first_id]["option_probabilities"])
            option = next(iter(probabilities))
            probabilities[option] = probabilities[option] - 0.01 if probabilities[option] > 0.01 else probabilities[option] + 0.01
            result[first_id] = {**result[first_id], "option_probabilities": probabilities}
            return result

        with self.assertRaisesRegex(AssertionError, "probability"):
            assert_packed_matches_oracle(
                broken, reference_batches()[0], request_id="req-1", scorer=deterministic_scorer
            )

    def test_detects_candidate_set_mismatch(self):
        def broken(batch, *, request_id, scorer):
            result = score_shared_state(batch, request_id=request_id, scorer=scorer)
            first_id = next(iter(result))
            probabilities = dict(result[first_id]["option_probabilities"])
            probabilities.pop(next(iter(probabilities)))
            result[first_id] = {**result[first_id], "option_probabilities": probabilities}
            return result

        with self.assertRaisesRegex(AssertionError, "candidate sets differ"):
            assert_packed_matches_oracle(
                broken, reference_batches()[1], request_id="req-1", scorer=deterministic_scorer
            )

    def test_quantized_tolerance_is_explicit_not_a_silent_default(self):
        # Simulates a BF16/quantized packed kernel: a small, bounded rounding
        # error on every probability, not a real correctness bug.
        def slightly_rounded(batch, *, request_id, scorer):
            result = score_shared_state(batch, request_id=request_id, scorer=scorer)
            rounded = {}
            for decision_id, response in result.items():
                probabilities = {name: round(value, 2) for name, value in response["option_probabilities"].items()}
                rounded[decision_id] = {**response, "option_probabilities": probabilities}
            return rounded

        batch = reference_batches()[0]
        with self.assertRaisesRegex(AssertionError, "probability"):
            assert_packed_matches_oracle(
                slightly_rounded, batch, request_id="req-1", scorer=balanced_scorer, **EXACT_TOLERANCE
            )
        assert_packed_matches_oracle(
            slightly_rounded, batch, request_id="req-1", scorer=balanced_scorer, **QUANTIZED_TOLERANCE
        )


if __name__ == "__main__":
    unittest.main()

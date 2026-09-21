import unittest

from reflexmodels import Choice, choice_response, fallback_gate


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.choice = Choice("Sistem durumu.", "Hangi yol?", {"small": "Küçük model", "large": "Büyük model"})

    def test_accepts_clear_prediction(self):
        response = choice_response(self.choice, request_id="r", probabilities={"small": 0.9, "large": 0.1})
        gate = fallback_gate(response, min_confidence=0.8, min_margin=0.5, max_normalized_entropy=0.8)
        self.assertTrue(gate["accepted"])
        self.assertFalse(gate["fallback_required"])

    def test_routes_ambiguous_prediction(self):
        response = choice_response(self.choice, request_id="r", probabilities={"small": 0.51, "large": 0.49})
        gate = fallback_gate(response, min_confidence=0.8, min_margin=0.2, max_normalized_entropy=0.8)
        self.assertTrue(gate["fallback_required"])
        self.assertIn("low_confidence", gate["reason_codes"])
        self.assertIn("low_margin", gate["reason_codes"])

    def test_rejects_invalid_thresholds(self):
        response = choice_response(self.choice, request_id="r", probabilities={"small": 0.9, "large": 0.1})
        with self.assertRaisesRegex(ValueError, "thresholds"):
            fallback_gate(response, min_confidence=0.0)

    def test_routes_when_answerability_is_low(self):
        response = choice_response(self.choice, request_id="r", probabilities={"small": 0.9, "large": 0.1})
        gate = fallback_gate(
            response, min_confidence=0.8, answerability_probability=0.2, min_answerability=0.7
        )
        self.assertTrue(gate["fallback_required"])
        self.assertIn("insufficient_evidence", gate["reason_codes"])

    def test_requires_both_answerability_values(self):
        response = choice_response(self.choice, request_id="r", probabilities={"small": 0.9, "large": 0.1})
        with self.assertRaisesRegex(ValueError, "together"):
            fallback_gate(response, min_confidence=0.8, answerability_probability=0.5)


if __name__ == "__main__":
    unittest.main()

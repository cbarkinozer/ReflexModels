import unittest

from reflexmodels import Binary, Choice, SharedStateDecisions, score_decision, score_shared_state, softmax


class ScoringTests(unittest.TestCase):
    def test_scores_isolated_choice_branches_into_typed_probabilities(self):
        request = Choice("Kullanıcı dosya arıyor.", "Hangi araç?", {"files": "Dosyaları ara", "web": "Web'de ara"})
        seen = []
        def scorer(branch):
            seen.append(branch)
            return 2.0 if "Dosyaları ara" in branch else 0.0
        result = score_decision(request, request_id="r-1", scorer=scorer)
        self.assertEqual(result["selected_option"], "files")
        self.assertEqual(len(seen), 2)
        self.assertTrue(all("<STATE>" in branch and "<DECISION>" in branch for branch in seen))

    def test_shared_state_can_score_multiple_typed_decisions(self):
        state = "Sistem güvenli durumda."
        batch = SharedStateDecisions(state, {"tool": Choice(state, "Hangi araç?", {"a": "A", "b": "B"}), "continue": Binary(state, "Devam mı?")})
        results = score_shared_state(batch, request_id="r-2", scorer=lambda _: 0.0, workers=2)
        self.assertEqual(set(results), {"tool", "continue"})
        self.assertEqual(results["continue"]["decision_type"], "binary")
        self.assertAlmostEqual(sum(softmax({"a": 1.0, "b": 1.0}).values()), 1.0)


if __name__ == "__main__":
    unittest.main()

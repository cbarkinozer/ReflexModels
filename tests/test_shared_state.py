import unittest

from reflexmodels import Binary, Choice, SharedStateDecisions, batch_request_to_dict, decision_response


class SharedStateTests(unittest.TestCase):
    def test_batch_sends_shared_state_once_and_supports_binary(self):
        state = "Kullanıcı dosya arıyor."
        choice = Choice(state, "Hangi araç?", {"files": "Dosyaları ara", "web": "Web'de ara"})
        batch = SharedStateDecisions(state, {"tool": choice, "continue": Binary(state, "Devam edilsin mi?")})
        payload = batch_request_to_dict(batch, request_id="req-2")
        self.assertEqual(payload["state"], state)
        self.assertEqual(len(payload["decisions"]), 2)
        self.assertNotIn("state", payload["decisions"][0])
        response = decision_response(batch.decisions["continue"], request_id="req-2", probabilities={"yes": 0.7, "no": 0.3})
        self.assertEqual(response["decision_type"], "binary")
        self.assertEqual(response["selected_option"], "yes")

    def test_rejects_decision_with_other_state(self):
        with self.assertRaisesRegex(ValueError, "shared state"):
            SharedStateDecisions("durum", {"wrong": Binary("başka durum", "Devam mı?")})


if __name__ == "__main__":
    unittest.main()

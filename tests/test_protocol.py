import json
import unittest

from reflexmodels import Choice, choice_response, request_to_dict, request_to_json, response_to_json


class ProtocolTests(unittest.TestCase):
    def test_english_binary_candidates_use_english_text(self):
        from reflexmodels import Binary, request_to_dict
        request = Binary(state="The service is healthy.", question="Is it healthy?", language="en")
        self.assertEqual(request_to_dict(request, request_id="en-1")["options"], {"yes": "Yes", "no": "No"})

    def setUp(self):
        self.request = Choice("Kullanıcı dosya arıyor.", "Hangi araç?", {"files": "Dosyaları ara", "web": "Web'de ara"})

    def test_request_is_turkish_first_json(self):
        payload = request_to_dict(self.request, request_id="req-1")
        self.assertEqual(payload["protocol"], "reflex.decision.v1")
        self.assertEqual(payload["options"]["files"], "Dosyaları ara")
        self.assertEqual(json.loads(request_to_json(self.request, request_id="req-1"))["request_id"], "req-1")

    def test_response_can_only_select_supplied_option(self):
        response = choice_response(self.request, request_id="req-1", probabilities={"files": 0.8, "web": 0.2})
        self.assertEqual(response["selected_option"], "files")
        self.assertEqual(json.loads(response_to_json(response))["confidence"], 0.8)
        with self.assertRaisesRegex(ValueError, "exactly"):
            choice_response(self.request, request_id="req-1", probabilities={"files": 1.0, "invented": 0.0})

    def test_response_rejects_free_text_fields(self):
        response = choice_response(self.request, request_id="req-1", probabilities={"files": 0.5, "web": 0.5})
        response["explanation"] = "uydurulmuş metin"
        with self.assertRaisesRegex(ValueError, "does not match"):
            response_to_json(response)


if __name__ == "__main__":
    unittest.main()

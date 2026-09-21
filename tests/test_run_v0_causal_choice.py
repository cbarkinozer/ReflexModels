import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_v0_causal_choice.py"
SPEC = importlib.util.spec_from_file_location("run_v0_causal_choice", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class V0CausalChoiceRunnerTests(unittest.TestCase):
    def test_loads_runtime_choice_request(self):
        payload = {
            "request_id": "r-1", "language": "tr", "state": "Durum", "question": "Hangisi?",
            "options": {"a": "Bir", "b": "İki"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            request_id, request = MODULE.load_request(path)
        self.assertEqual(request_id, "r-1")
        self.assertEqual(request.options["b"], "İki")

    def test_rejects_missing_request_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps({"state": "D", "question": "S", "options": {"a": "A", "b": "B"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "request_id"):
                MODULE.load_request(path)

    def test_loads_binary_request_without_fixed_options(self):
        payload = {"request_id": "r-2", "decision_type": "binary", "state": "Durum", "question": "Doğru mu?"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            _, request = MODULE.load_request(path)
        self.assertEqual(type(request).__name__, "Binary")


if __name__ == "__main__":
    unittest.main()

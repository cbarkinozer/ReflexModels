import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_v1_decision.py"
SPEC = importlib.util.spec_from_file_location("run_v1_decision", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

from reflexmodels.types import Choice


class V1DecisionRunnerTests(unittest.TestCase):
    def test_bounded_response_routes_insufficient_evidence(self):
        request = Choice(state="Durum", question="Hangisi?", options={"a": "Bir", "b": "İki"})
        output = MODULE.build_output(
            request, "r-1", {"a": 3.0, "b": 1.0}, 0.2,
            temperature=1.0, answerability_threshold=0.7,
        )
        self.assertEqual(output["decision"]["selected_option"], "a")
        self.assertTrue(output["routing"]["fallback_required"])
        self.assertIn("insufficient_evidence", output["routing"]["reason_codes"])
        self.assertNotIn("explanation", output["decision"])


if __name__ == "__main__":
    unittest.main()

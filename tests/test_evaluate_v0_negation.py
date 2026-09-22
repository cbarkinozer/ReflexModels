import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_v0_negation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_v0_negation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class V0NegationEvaluationTests(unittest.TestCase):
    def test_pair_accuracy_requires_both_sides_correct(self):
        records = [
            {"id": "a", "pair_id": "p", "state": "Evet", "question": "Olur mu?", "label": True},
            {"id": "b", "pair_id": "p", "state": "Hayır", "question": "Olur mu?", "label": False},
        ]

        def scorer(request, request_id):
            p_yes = 0.9 if request_id == "a" else 0.1
            return {"selected_option": "yes" if p_yes > 0.5 else "no",
                    "option_probabilities": {"no": 1 - p_yes, "yes": p_yes}}

        predictions, metrics = MODULE.evaluate_records(records, scorer)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["pair_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_v1_massive_pilot.py"
SPEC = importlib.util.spec_from_file_location("prepare_v1_massive_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def source(identifier, text, label):
    return {"id": identifier, "source_id": identifier, "origin": "translated", "text": text, "label_name": label}


class V1MassivePilotTests(unittest.TestCase):
    def test_positive_and_insufficient_pairs_change_only_options_and_target(self):
        descriptions = {name: name for name in ("alarm_set", "alarm_query", "alarm_remove", "music_query", "news_query")}
        src = source("a", "Alarm kur", "alarm_set")
        positive = MODULE.option_record(src, descriptions, split="train", answerable=True, seed=42)
        negative = MODULE.option_record(src, descriptions, split="train", answerable=False, seed=42)
        self.assertEqual(positive["state"], negative["state"])
        self.assertIn("alarm_set", positive["options"])
        self.assertNotIn("alarm_set", negative["options"])
        self.assertIsNone(negative["correct_option"])

    def test_validation_text_does_not_overlap_selected_train(self):
        descriptions = {name: name for name in ("alarm_set", "alarm_query", "music_query", "news_query", "qa_maths")}
        train = [source("a", "Alarm kur", "alarm_set")]
        validation = [source("b", "ALARM KUR", "alarm_set"), source("c", "Müzik sor", "music_query")]
        rows = MODULE.build_records(train, validation, descriptions, train_limit=1, validation_limit=1, seed=42)
        validation_states = {row["state"] for row in rows if row["split"] == "validation"}
        self.assertEqual(validation_states, {"Müzik sor"})
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()

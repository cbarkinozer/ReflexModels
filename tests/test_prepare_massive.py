import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_massive.py"
SPEC = importlib.util.spec_from_file_location("prepare_massive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def record(identifier, partition, intent, utterance):
    return {"id": identifier, "partition": partition, "intent": intent, "utt": utterance}


class MassivePreparationTests(unittest.TestCase):
    def test_prepares_shared_label_ids_and_preserves_provenance(self):
        prepared = MODULE.prepare({
            "tr": [record("1", "train", "alarm_set", "Alarm kur"), record("2", "test", "weather", "Hava nasıl?")],
            "en": [record("1", "train", "alarm_set", "Set an alarm"), record("2", "test", "weather", "What is the weather?")],
        })
        self.assertEqual(prepared["labels"], [{"id": 0, "name": "alarm_set"}, {"id": 1, "name": "weather"}])
        self.assertEqual(prepared["tr_train"][0]["origin"], "translated")
        self.assertEqual(prepared["tr_train"][0]["source_id"], "1")
        self.assertEqual(prepared["en_test"][0]["label_name"], "weather")

    def test_drops_labels_not_shared_by_both_languages(self):
        prepared = MODULE.prepare({
            "tr": [record("1", "train", "shared", "ortak"), record("2", "train", "tr_only", "yalnız")],
            "en": [record("1", "train", "shared", "shared")],
        })
        self.assertEqual(len(prepared["tr_train"]), 1)
        self.assertEqual(prepared["labels"], [{"id": 0, "name": "shared"}])


if __name__ == "__main__":
    unittest.main()

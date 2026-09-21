import unittest

from reflexmodels.split_policy import apply_overlap_policy


class SplitPolicyTests(unittest.TestCase):
    def test_sensitivity_removes_train_dev_matches_but_never_test(self):
        splits = {
            "train": [{"id": "a", "text": " Merhaba "}, {"id": "b", "text": "başka"}],
            "dev": [{"id": "c", "text": "MERHABA"}],
            "test": [{"id": "d", "text": "merhaba"}],
        }
        output, removed = apply_overlap_policy(splits, policy="drop_train_dev_overlapping_test")
        self.assertEqual([item["id"] for item in output["train"]], ["b"])
        self.assertEqual(output["dev"], [])
        self.assertEqual(output["test"], splits["test"])
        self.assertEqual(removed, {"train": 1, "dev": 1, "test": 0})

    def test_report_only_retains_records(self):
        splits = {name: [{"text": "same"}] for name in ("train", "dev", "test")}
        output, removed = apply_overlap_policy(splits, policy="report_only")
        self.assertEqual(output, splits)
        self.assertEqual(removed, {"train": 0, "dev": 0, "test": 0})


if __name__ == "__main__":
    unittest.main()

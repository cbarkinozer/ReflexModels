import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_results_registry.py"
SPEC = importlib.util.spec_from_file_location("validate_results_registry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class LinkedResultPathsTests(unittest.TestCase):
    def test_extracts_markdown_links_into_results(self):
        text = "| [`a`](results/a/result.json) | ok |\n| plain text row |\n| [`b`](results/b.json) | ok |\n"
        self.assertEqual(MODULE.linked_result_paths(text), ["results/a/result.json", "results/b.json"])

    def test_ignores_links_outside_results(self):
        text = "[docs](PLAN.md) and [x](results/x.json)"
        self.assertEqual(MODULE.linked_result_paths(text), ["results/x.json"])


class ValidateLinksTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        (self.root / "results").mkdir()

    def test_accepts_existing_valid_json(self):
        path = self.root / "results" / "a.json"
        path.write_text(json.dumps({"run_id": "a"}), encoding="utf-8")
        MODULE.validate_links(["results/a.json"], root=self.root)

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            MODULE.validate_links(["results/missing.json"], root=self.root)

    def test_rejects_invalid_json(self):
        path = self.root / "results" / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            MODULE.validate_links(["results/bad.json"], root=self.root)

    def test_rejects_empty_object(self):
        path = self.root / "results" / "empty.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            MODULE.validate_links(["results/empty.json"], root=self.root)

    def test_rejects_duplicate_links(self):
        path = self.root / "results" / "a.json"
        path.write_text(json.dumps({"run_id": "a"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "more than once"):
            MODULE.validate_links(["results/a.json", "results/a.json"], root=self.root)

    def test_rejects_path_escaping_results_via_dotdot(self):
        secret = self.root / "secret.json"
        secret.write_text(json.dumps({"leak": True}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes results/"):
            MODULE.validate_links(["results/../secret.json"], root=self.root)


class FindUndocumentedResultsTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        (self.root / "results").mkdir()

    def test_flags_result_json_not_linked(self):
        run_dir = self.root / "results" / "some_run"
        run_dir.mkdir()
        (run_dir / "result.json").write_text("{}", encoding="utf-8")
        undocumented = MODULE.find_undocumented_results(root=self.root, linked_paths=[])
        self.assertEqual(undocumented, ["results/some_run/result.json"])

    def test_does_not_flag_linked_result(self):
        run_dir = self.root / "results" / "some_run"
        run_dir.mkdir()
        (run_dir / "result.json").write_text("{}", encoding="utf-8")
        undocumented = MODULE.find_undocumented_results(
            root=self.root, linked_paths=["results/some_run/result.json"]
        )
        self.assertEqual(undocumented, [])

    def test_exempts_smoke_and_eval_artifacts(self):
        for name in ("run_smoke", "run_eval"):
            run_dir = self.root / "results" / name
            run_dir.mkdir()
            (run_dir / "result.json").write_text("{}", encoding="utf-8")
        undocumented = MODULE.find_undocumented_results(root=self.root, linked_paths=[])
        self.assertEqual(undocumented, [])


class RealRepositoryRegistryTests(unittest.TestCase):
    def test_current_results_md_is_internally_consistent(self):
        results_md = Path(__file__).parents[1] / "RESULTS.md"
        text = results_md.read_text(encoding="utf-8")
        paths = MODULE.linked_result_paths(text)
        MODULE.validate_links(paths, root=MODULE.ROOT)
        undocumented = MODULE.find_undocumented_results(root=MODULE.ROOT, linked_paths=paths)
        self.assertEqual(undocumented, [])


if __name__ == "__main__":
    unittest.main()

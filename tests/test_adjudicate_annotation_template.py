import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "adjudicate_annotation_template.py"
SPEC = importlib.util.spec_from_file_location("adjudicate_annotation_template", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

HEADER = "\t".join(MODULE.LINT.REQUIRED_COLUMNS)
TOOL_OPTIONS = json.dumps({"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"}, ensure_ascii=False)


def row(
    item_id, task_family="tool_routing", state="", question="", options_json="",
    a1="", a2="", adjudicated="", notes="",
    annotator_1_id="", annotator_2_id="", adjudicator_id="",
    date_1="", date_2="", adjudication_date="", guideline_version="",
):
    return "\t".join([
        item_id, task_family, state, question, options_json, a1, a2, adjudicated, notes,
        annotator_1_id, annotator_2_id, adjudicator_id, date_1, date_2, adjudication_date, guideline_version,
    ])


def write(directory: Path, *rows: str) -> Path:
    path = directory / "items.tsv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def annotated(item_id, *, task_family="tool_routing", options_json=TOOL_OPTIONS, a1="filesystem", a2="filesystem",
              adjudicated="", notes=""):
    kwargs = dict(
        task_family=task_family, state="Durum.", question="Hangi araç?", options_json=options_json,
        a1=a1, a2=a2, notes=notes, annotator_1_id="ann1", annotator_2_id="ann2",
        date_1="2026-01-01", date_2="2026-01-02", guideline_version="v1",
    )
    if adjudicated:
        kwargs.update(adjudicated=adjudicated, adjudicator_id="adj1", adjudication_date="2026-01-03")
    return row(item_id, **kwargs)


class CohensKappaTests(unittest.TestCase):
    def test_perfect_agreement_with_varied_labels(self):
        kappa = MODULE.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
        self.assertAlmostEqual(kappa, 1.0)

    def test_single_category_agreement_is_undefined_not_perfect(self):
        kappa = MODULE.cohens_kappa(["a", "a", "a"], ["a", "a", "a"])
        self.assertIsNone(kappa)

    def test_chance_level_agreement_is_near_zero(self):
        annotation_1 = ["a", "b"] * 20
        annotation_2 = ["a", "a", "b", "b"] * 10
        kappa = MODULE.cohens_kappa(annotation_1, annotation_2)
        self.assertLess(abs(kappa), 0.3)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            MODULE.cohens_kappa(["a"], ["a", "b"])


class AgreementByFamilyTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def test_only_counts_double_annotated_and_adjudicated_rows(self):
        path = write(
            self.directory,
            row("TR-1", task_family="intent"),  # blank, excluded
            row("TR-2", task_family="intent", state="D", question="S?", options_json=TOOL_OPTIONS),  # drafted
            annotated("TR-3", a1="filesystem", a2="filesystem"),
            annotated("TR-4", a1="filesystem", a2="web", adjudicated="filesystem"),
        )
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        stats = MODULE.agreement_by_family(rows)
        self.assertEqual(stats["tool_routing"]["n"], 2)
        self.assertAlmostEqual(stats["tool_routing"]["raw_agreement"], 0.5)

    def test_surfaces_kappa_defined_flag(self):
        path = write(self.directory, annotated("TR-1", a1="filesystem", a2="filesystem"))
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        stats = MODULE.agreement_by_family(rows)
        self.assertFalse(stats["tool_routing"]["kappa_defined"])
        self.assertIsNone(stats["tool_routing"]["cohens_kappa"])

    def test_empty_when_no_double_annotated_rows(self):
        path = write(self.directory, row("TR-1", task_family="intent"))
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        self.assertEqual(MODULE.agreement_by_family(rows), {})


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def test_selects_only_adjudicated_rows(self):
        path = write(
            self.directory,
            annotated("TR-1", adjudicated="filesystem"),
            annotated("TR-2", a1="filesystem", a2="web"),
        )
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        adjudicated = MODULE.select_adjudicated(rows)
        self.assertEqual([row["item_id"] for row in adjudicated], ["TR-1"])

    def test_frozen_schema_matches_v1_data_contract(self):
        path = write(self.directory, annotated("TR-1", adjudicated="filesystem", notes="clear case"))
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        records = MODULE.to_frozen_records(MODULE.select_adjudicated(rows))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["split"], "test")
        self.assertEqual(record["options"], {"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"})
        self.assertEqual(record["correct_option"], "filesystem")
        self.assertIs(record["answerable"], True)
        self.assertEqual(record["source"], "reflexbench-tr-tool_routing")
        self.assertEqual(record["provenance"]["annotator_1_id"], "ann1")
        self.assertEqual(record["provenance"]["notes"], "clear case")
        validator = MODULE._load_module(
            "validate_v1_decision_data", MODULE.ROOT / "scripts" / "validate_v1_decision_data.py"
        )
        validator.validate(records, allowed_splits=frozenset({"test"}))

    def test_answerability_with_real_option_is_answerable_true(self):
        path = write(self.directory, annotated(
            "TR-1", task_family="answerability", a1="filesystem", a2="filesystem", adjudicated="filesystem",
        ))
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        records = MODULE.to_frozen_records(MODULE.select_adjudicated(rows))
        self.assertIs(records[0]["answerable"], True)
        self.assertEqual(records[0]["correct_option"], "filesystem")
        self.assertEqual(records[0]["options"], {"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"})

    def test_answerability_with_insufficient_label_is_answerable_false(self):
        path = write(self.directory, annotated(
            "TR-1", task_family="answerability", a1="insufficient", a2="insufficient", adjudicated="insufficient",
        ))
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        records = MODULE.to_frozen_records(MODULE.select_adjudicated(rows))
        self.assertIs(records[0]["answerable"], False)
        self.assertIsNone(records[0]["correct_option"])
        self.assertEqual(records[0]["options"], {"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"})
        validator = MODULE._load_module(
            "validate_v1_decision_data", MODULE.ROOT / "scripts" / "validate_v1_decision_data.py"
        )
        validator.validate(records, allowed_splits=frozenset({"test"}))

    def test_relevance_is_ingested_as_choice_record(self):
        options = json.dumps({"relevant": "İlgili", "not_relevant": "İlgisiz"}, ensure_ascii=False)
        path = write(self.directory, annotated(
            "TR-1", task_family="relevance", options_json=options,
            a1="relevant", a2="relevant", adjudicated="relevant",
        ))
        rows = MODULE.LINT.load_rows(path)
        MODULE.LINT.validate(rows)
        records = MODULE.to_frozen_records(MODULE.select_adjudicated(rows))
        self.assertEqual(records[0]["correct_option"], "relevant")
        self.assertIs(records[0]["answerable"], True)

    def test_write_frozen_rejects_empty_records(self):
        with self.assertRaisesRegex(ValueError, "no adjudicated"):
            MODULE.write_frozen([], self.directory / "out.jsonl")

    def test_write_frozen_round_trips_as_jsonl(self):
        records = [{"id": "TR-1", "language": "tr"}]
        output = self.directory / "out.jsonl"
        MODULE.write_frozen(records, output)
        loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(loaded, records)

    def test_write_frozen_refuses_to_overwrite_existing_file(self):
        output = self.directory / "out.jsonl"
        output.write_text('{"already": "here"}\n', encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            MODULE.write_frozen([{"id": "TR-1"}], output)
        self.assertEqual(output.read_text(encoding="utf-8"), '{"already": "here"}\n')


if __name__ == "__main__":
    unittest.main()

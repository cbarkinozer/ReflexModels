import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "lint_annotation_template.py"
SPEC = importlib.util.spec_from_file_location("lint_annotation_template", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

HEADER = "\t".join(MODULE.REQUIRED_COLUMNS)
TOOL_OPTIONS = json.dumps({"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"}, ensure_ascii=False)
ANSWERABILITY_OPTIONS = json.dumps({"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"}, ensure_ascii=False)


def row(
    item_id, task_family="intent", state="", question="", options_json="",
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


def annotated(item_id, *, task_family="tool_routing", options_json=TOOL_OPTIONS, a1="filesystem", a2="filesystem", adjudicated=""):
    kwargs = dict(
        task_family=task_family, state="Durum.", question="Hangi araç?", options_json=options_json,
        a1=a1, a2=a2, annotator_1_id="ann1", annotator_2_id="ann2", date_1="2026-01-01", date_2="2026-01-02",
        guideline_version="v1",
    )
    if adjudicated:
        kwargs.update(adjudicated=adjudicated, adjudicator_id="adj1", adjudication_date="2026-01-03")
    return row(item_id, **kwargs)


class LintAnnotationTemplateTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def test_loads_real_template_file(self):
        real = Path(__file__).parents[1] / "benchmarks/reflexbench_tr/annotation_templates/items.tsv"
        rows = MODULE.load_rows(real)
        MODULE.validate(rows)
        self.assertGreater(len(rows), 0)

    def test_accepts_blank_row(self):
        path = write(self.directory, row("TR-1"))
        MODULE.validate(MODULE.load_rows(path))

    def test_drafted_row_requires_options_json(self):
        path = write(self.directory, row("TR-1", task_family="tool_routing", state="D", question="S?"))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_accepts_drafted_row_with_options(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
        ))
        rows = MODULE.load_rows(path)
        MODULE.validate(rows)
        self.assertEqual(MODULE.row_stage(rows[0]), "drafted")

    def test_rejects_annotation_not_in_options(self):
        path = write(self.directory, annotated("TR-1", a1="not_an_option"))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_accepts_double_annotated_and_adjudicated(self):
        path = write(
            self.directory,
            annotated("TR-1"),
            annotated("TR-2", adjudicated="filesystem"),
        )
        rows = MODULE.load_rows(path)
        MODULE.validate(rows)
        self.assertEqual(MODULE.row_stage(rows[0]), "double_annotated")
        self.assertEqual(MODULE.row_stage(rows[1]), "adjudicated")

    def test_rejects_annotation_without_annotator_id(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
            a1="filesystem", a2="filesystem",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_adjudication_without_adjudicator_id(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
            a1="filesystem", a2="filesystem", adjudicated="filesystem",
            annotator_1_id="a1", annotator_2_id="a2", date_1="2026-01-01", date_2="2026-01-02", guideline_version="v1",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_answerability_accepts_insufficient_label(self):
        path = write(self.directory, annotated(
            "TR-1", task_family="answerability", options_json=ANSWERABILITY_OPTIONS,
            a1="insufficient", a2="insufficient", adjudicated="insufficient",
        ))
        rows = MODULE.load_rows(path)
        MODULE.validate(rows)
        self.assertEqual(MODULE.row_stage(rows[0]), "adjudicated")

    def test_answerability_accepts_a_real_option_as_the_answer(self):
        path = write(self.directory, annotated(
            "TR-1", task_family="answerability", options_json=ANSWERABILITY_OPTIONS,
            a1="filesystem", a2="filesystem", adjudicated="filesystem",
        ))
        rows = MODULE.load_rows(path)
        MODULE.validate(rows)
        self.assertEqual(MODULE.row_stage(rows[0]), "adjudicated")

    def test_answerability_rejects_insufficient_as_an_option_key(self):
        bad_options = json.dumps({"insufficient": "x", "web": "y"})
        path = write(self.directory, row(
            "TR-1", task_family="answerability", state="D", question="S?", options_json=bad_options,
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_non_answerability_family_rejects_insufficient_label(self):
        path = write(self.directory, annotated("TR-1", task_family="tool_routing", a1="insufficient", a2="insufficient"))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_relevance_is_choice_shaped(self):
        options = json.dumps({"relevant": "İlgili", "not_relevant": "İlgisiz"}, ensure_ascii=False)
        path = write(self.directory, annotated(
            "TR-1", task_family="relevance", options_json=options, a1="relevant", a2="relevant", adjudicated="relevant",
        ))
        rows = MODULE.load_rows(path)
        MODULE.validate(rows)
        self.assertEqual(MODULE.row_stage(rows[0]), "adjudicated")

    def test_rejects_duplicate_item_id(self):
        path = write(self.directory, row("TR-1"), row("TR-1"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_unknown_task_family(self):
        path = write(self.directory, row("TR-1", task_family="not_a_family"))
        with self.assertRaisesRegex(ValueError, "task_family"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_single_sided_annotation(self):
        path = write(self.directory, annotated("TR-1", a2=""))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_adjudication_without_both_annotations(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
            adjudicated="filesystem", adjudicator_id="adj1", adjudication_date="2026-01-01",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_annotation_without_state_or_question(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", options_json=TOOL_OPTIONS, a1="filesystem", a2="filesystem",
            annotator_1_id="a1", annotator_2_id="a2", date_1="2026-01-01", date_2="2026-01-02", guideline_version="v1",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_wrong_header(self):
        path = self.directory / "bad.tsv"
        path.write_text("id\tfamily\n1\tintent\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "header"):
            MODULE.load_rows(path)

    def test_parse_options_rejects_single_option(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            MODULE.parse_options(json.dumps({"a": "only one"}), item_id="TR-1")

    def test_parse_options_rejects_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            MODULE.parse_options("not json", item_id="TR-1")

    def test_rejects_dangling_annotator_id_on_blank_annotation(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
            annotator_1_id="ann1",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_dangling_annotation_date_on_blank_annotation(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
            date_1="2026-01-01",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_dangling_adjudicator_id_without_adjudicated_label(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
            a1="filesystem", a2="filesystem", annotator_1_id="a1", annotator_2_id="a2",
            date_1="2026-01-01", date_2="2026-01-02", guideline_version="v1", adjudicator_id="adj1",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_dangling_guideline_version_without_annotations(self):
        path = write(self.directory, row(
            "TR-1", task_family="tool_routing", state="D", question="S?", options_json=TOOL_OPTIONS,
            guideline_version="v1",
        ))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MODULE.validate(MODULE.load_rows(path))

    def test_rejects_non_iso_annotation_date(self):
        path = write(self.directory, annotated("TR-1"))
        rows = MODULE.load_rows(path)
        rows[0]["annotation_date_1"] = "01/01/2026"
        self.assertEqual(MODULE.row_stage(rows[0]), "invalid")

    def test_rejects_non_iso_adjudication_date(self):
        path = write(self.directory, annotated("TR-1", adjudicated="filesystem"))
        rows = MODULE.load_rows(path)
        rows[0]["adjudication_date"] = "not-a-date"
        self.assertEqual(MODULE.row_stage(rows[0]), "invalid")

    def test_summarize_counts_stages_per_family(self):
        path = write(
            self.directory,
            row("TR-1", task_family="intent"),
            row("TR-2", task_family="intent", state="D", question="S?", options_json=TOOL_OPTIONS),
            annotated("TR-3", adjudicated="filesystem"),
        )
        rows = MODULE.load_rows(path)
        MODULE.validate(rows)
        summary = MODULE.summarize(rows)
        self.assertEqual(summary["intent"], {"blank": 1, "drafted": 1, "double_annotated": 0, "adjudicated": 0})
        self.assertEqual(summary["tool_routing"], {"blank": 0, "drafted": 0, "double_annotated": 0, "adjudicated": 1})


if __name__ == "__main__":
    unittest.main()

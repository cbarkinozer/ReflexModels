import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_v1_decision_model.py"
SPEC = importlib.util.spec_from_file_location("train_v1_decision_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [(ord(character) % 255) + 1 for character in text]


class FakeConfig:
    hidden_size = 4


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.embedding = nn.Embedding(256, 4)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        return SimpleNamespace(hidden_states=[self.embedding(input_ids)])


def make_example(record_id, *, answerable=True, correct_option="a", options=None, split="validation"):
    return MODULE.DecisionExample(
        record_id=record_id,
        split=split,
        state="Durum.",
        question="Ne yapılmalı?",
        options=tuple((options or {"a": "A seçeneği", "b": "B seçeneği"}).items()),
        answerable=answerable,
        correct_option=correct_option if answerable else None,
    )


class LoadExamplesTests(unittest.TestCase):
    def _write(self, records):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "data.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
        return path

    def _record(self, identifier, split, **overrides):
        value = {
            "id": identifier, "language": "tr", "origin": "native", "source": "curated-v1",
            "split": split, "state": "Durum.", "question": "Ne yapılmalı?",
            "options": {"a": "A seçeneği", "b": "B seçeneği"},
            "answerable": True, "correct_option": "a",
        }
        value.update(overrides)
        return value

    def test_splits_train_and_validation_records(self):
        path = self._write([
            self._record("tr-1", "train"), self._record("tr-2", "validation"),
            self._record("tr-3", "validation", answerable=False, correct_option=None),
        ])
        train, validation = MODULE.load_examples(path)
        self.assertEqual([e.record_id for e in train], ["tr-1"])
        self.assertEqual([e.record_id for e in validation], ["tr-2", "tr-3"])

    def test_rejects_validation_without_insufficient_examples(self):
        path = self._write([self._record("tr-1", "train"), self._record("tr-2", "validation")])
        with self.assertRaisesRegex(ValueError, "answerable and unanswerable"):
            MODULE.load_examples(path)

    def test_rejects_frozen_test_split(self):
        path = self._write([self._record("tr-1", "train"), self._record("tr-2", "test")])
        with self.assertRaisesRegex(ValueError, "split"):
            MODULE.load_examples(path)

    def test_rejects_missing_validation_split(self):
        path = self._write([self._record("tr-1", "train"), self._record("tr-2", "train")])
        with self.assertRaisesRegex(ValueError, "validation"):
            MODULE.load_examples(path)


class BatchingTests(unittest.TestCase):
    def test_groups_variable_option_counts_per_record(self):
        examples = [
            make_example("tr-1", options={"a": "A", "b": "B"}),
            make_example("tr-2", options={"x": "X", "y": "Y", "z": "Z"}, correct_option="z"),
        ]
        batch = MODULE.make_batch(examples, FakeTokenizer(), max_length=32)
        self.assertEqual(batch.record_option_ranges, [(0, 2), (2, 5)])
        self.assertEqual(batch.record_option_names, [["a", "b"], ["x", "y", "z"]])
        self.assertEqual(batch.target_index, [0, 2])
        self.assertEqual(batch.branch_input_ids.shape[0], 5)
        self.assertTrue(torch.equal(batch.decision_mask, torch.tensor([True, True])))

    def test_unanswerable_record_gets_negative_target_and_false_mask(self):
        examples = [make_example("tr-1", answerable=False)]
        batch = MODULE.make_batch(examples, FakeTokenizer(), max_length=32)
        self.assertEqual(batch.target_index, [-1])
        self.assertFalse(bool(batch.decision_mask[0]))

    def test_answerability_inputs_have_one_row_per_record(self):
        examples = [make_example("tr-1"), make_example("tr-2", answerable=False)]
        batch = MODULE.make_batch(examples, FakeTokenizer(), max_length=32)
        self.assertEqual(batch.answerable_input_ids.shape[0], 2)
        self.assertTrue(torch.equal(batch.answerable_labels, torch.tensor([1.0, 0.0])))

    def test_rejects_correct_option_not_among_options(self):
        example = make_example("tr-1", correct_option="missing")
        with self.assertRaisesRegex(ValueError, "correct_option"):
            MODULE.make_batch([example], FakeTokenizer(), max_length=32)


class LossMaskingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MODULE.DecisionModel(FakeBackbone())

    def test_decision_loss_ignores_unanswerable_records(self):
        answerable = make_example("tr-1", answerable=True, correct_option="a")
        unanswerable_low = make_example("tr-2", answerable=False, options={"a": "Farkli metin", "b": "Baska"})
        unanswerable_high = make_example(
            "tr-2", answerable=False, options={"a": "COMPLETELY DIFFERENT LONG TEXT HERE", "b": "Baska"}
        )
        batch_low = MODULE.make_batch([answerable, unanswerable_low], FakeTokenizer(), max_length=32)
        batch_high = MODULE.make_batch([answerable, unanswerable_high], FakeTokenizer(), max_length=32)
        _, decision_loss_low, _, count_low = MODULE.compute_losses(self.model, batch_low)
        _, decision_loss_high, _, count_high = MODULE.compute_losses(self.model, batch_high)
        self.assertEqual(count_low, 1)
        self.assertEqual(count_high, 1)
        self.assertAlmostEqual(decision_loss_low, decision_loss_high, places=5)

    def test_decision_loss_is_zero_when_batch_has_no_answerable_records(self):
        batch = MODULE.make_batch([make_example("tr-1", answerable=False)], FakeTokenizer(), max_length=32)
        _, decision_loss, _, count = MODULE.compute_losses(self.model, batch)
        self.assertEqual(count, 0)
        self.assertEqual(decision_loss, 0.0)

    def test_answerability_loss_uses_every_record(self):
        examples = [make_example("tr-1", answerable=True), make_example("tr-2", answerable=False)]
        batch = MODULE.make_batch(examples, FakeTokenizer(), max_length=32)
        _, _, answerability_loss, _ = MODULE.compute_losses(self.model, batch)
        branch_hidden = self.model._pooled_hidden(
            batch.branch_input_ids, batch.branch_attention_mask, batch.branch_lengths
        )
        logits = self.model.answerability_logits(
            batch.answerable_input_ids, batch.answerable_attention_mask, batch.answerable_lengths,
            branch_hidden, batch.record_option_ranges,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.answerable_labels).item()
        self.assertAlmostEqual(answerability_loss, expected, places=5)

    def test_gradients_flow_to_both_heads(self):
        examples = [make_example("tr-1", answerable=True), make_example("tr-2", answerable=False)]
        batch = MODULE.make_batch(examples, FakeTokenizer(), max_length=32)
        loss, _, _, _ = MODULE.compute_losses(self.model, batch)
        loss.backward()
        self.assertIsNotNone(self.model.decision_head.weight.grad)
        self.assertIsNotNone(self.model.answerability_head.weight.grad)

    def test_answerability_changes_with_supplied_options(self):
        examples = [
            make_example("tr-1", options={"a": "A", "b": "B"}),
            make_example("tr-2", options={"a": "X", "b": "Y"}),
        ]
        batch = MODULE.make_batch(examples, FakeTokenizer(), max_length=32)
        with torch.no_grad():
            self.model.answerability_head.weight.zero_()
            self.model.answerability_head.weight[0, 4:] = 1.0
            self.model.answerability_head.bias.zero_()
        branch_hidden = self.model._pooled_hidden(
            batch.branch_input_ids, batch.branch_attention_mask, batch.branch_lengths
        )
        logits = self.model.answerability_logits(
            batch.answerable_input_ids, batch.answerable_attention_mask, batch.answerable_lengths,
            branch_hidden, batch.record_option_ranges,
        )
        self.assertNotEqual(float(logits[0]), float(logits[1]))

    def test_answerability_is_invariant_to_option_order(self):
        examples = [
            make_example("tr-1", options={"a": "A", "b": "B"}),
            make_example("tr-2", options={"b": "B", "a": "A"}),
        ]
        batch = MODULE.make_batch(examples, FakeTokenizer(), max_length=32)
        branch_hidden = self.model._pooled_hidden(
            batch.branch_input_ids, batch.branch_attention_mask, batch.branch_lengths
        )
        logits = self.model.answerability_logits(
            batch.answerable_input_ids, batch.answerable_attention_mask, batch.answerable_lengths,
            branch_hidden, batch.record_option_ranges,
        )
        self.assertAlmostEqual(float(logits[0]), float(logits[1]), places=6)

    def test_bfloat16_backbone_features_work_with_float32_heads(self):
        model = MODULE.DecisionModel(FakeBackbone().to(dtype=torch.bfloat16))
        batch = MODULE.make_batch([make_example("tr-1")], FakeTokenizer(), max_length=32)
        loss, _, _, _ = MODULE.compute_losses(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.answerability_head.weight.grad)

    def test_frozen_backbone_stays_in_eval_mode_during_head_training(self):
        model = MODULE.DecisionModel(FakeBackbone())
        model.backbone.requires_grad_(False)
        model.backbone_frozen = True
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.decision_head.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))


class CalibrationAndThresholdSelectionTests(unittest.TestCase):
    def test_fits_temperature_and_threshold_from_validation_only(self):
        validation_examples = [
            make_example("tr-1", answerable=True, correct_option="a"),
            make_example("tr-2", answerable=True, correct_option="b"),
            make_example("tr-3", answerable=False),
        ]
        score_maps = [{"a": 3.0, "b": 0.0}, {"a": 0.0, "b": 3.0}, {"a": 1.0, "b": 1.0}]
        answerable_probabilities = [0.9, 0.85, 0.2]
        result = MODULE.select_calibration_and_threshold(
            validation_examples, score_maps, answerable_probabilities, min_coverage=0.5
        )
        self.assertGreater(result["temperature"], 0)
        self.assertIn("threshold", result)
        self.assertGreaterEqual(result["coverage"], 0.5)

    def test_rejects_train_split_examples(self):
        validation_examples = [make_example("tr-1", split="train")]
        with self.assertRaisesRegex(ValueError, "split=validation"):
            MODULE.select_calibration_and_threshold(validation_examples, [{"a": 1.0, "b": 0.0}], [0.9])

    def test_rejects_misaligned_inputs(self):
        validation_examples = [make_example("tr-1")]
        with self.assertRaisesRegex(ValueError, "aligned"):
            MODULE.select_calibration_and_threshold(validation_examples, [{"a": 1.0}, {"b": 1.0}], [0.9])

    def test_temperature_fit_prefers_more_confident_correct_scaling(self):
        score_maps = [{"a": 5.0, "b": 0.0}] * 5
        correct_options = ["a"] * 5
        temperature = MODULE.fit_decision_temperature(score_maps, correct_options)
        metrics_at_fit = MODULE.decision_validation_metrics(score_maps, correct_options, temperature=temperature)
        metrics_at_ten = MODULE.decision_validation_metrics(score_maps, correct_options, temperature=10.0)
        self.assertLessEqual(metrics_at_fit["nll"], metrics_at_ten["nll"] + 1e-9)

    def test_decision_validation_metrics_reject_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            MODULE.decision_validation_metrics([{"a": 1.0}], ["a", "b"], temperature=1.0)


if __name__ == "__main__":
    unittest.main()

"""Real end-to-end CPU smoke test for scripts/train_v1_decision_model.py.

No network access and no repository assets are used: a tiny local GPT-2-style
base model and byte-level BPE tokenizer are built programmatically into a
pytest/unittest temp directory, alongside a handful of synthetic V1 JSONL
records. The actual CLI subprocess is invoked for one epoch, then the saved
backbone + heads are reloaded twice from disk to confirm the checkpoint is
self-contained and reproducible.

This never touches data/raw/models, results/, or any existing checkpoint.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "scripts" / "train_v1_decision_model.py"

import importlib.util

_SPEC = importlib.util.spec_from_file_location("train_v1_decision_model_smoke_target", SCRIPT)
TRAINER = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = TRAINER
_SPEC.loader.exec_module(TRAINER)

TRAIN_CORPUS = [
    "Durum: Kullanici config.yaml dosyasini incelemek istiyor.",
    "Hangi arac kullanilmali?",
    "Yerel dosyalari oku",
    "Internette ara",
    "Durum: Kullanici hava durumunu ogrenmek istiyor ama konum belirtilmemis.",
    "Hangi sehir icin?",
    "Istanbul",
    "Ankara",
]

RECORDS = [
    {
        "id": "tr-train-1", "language": "tr", "origin": "native", "source": "smoke", "split": "train",
        "state": "Durum: Kullanici config.yaml dosyasini incelemek istiyor.",
        "question": "Hangi arac kullanilmali?",
        "options": {"filesystem": "Yerel dosyalari oku", "web": "Internette ara"},
        "answerable": True, "correct_option": "filesystem",
    },
    {
        "id": "tr-train-2", "language": "tr", "origin": "native", "source": "smoke", "split": "train",
        "state": "Durum: Kullanici hava durumunu ogrenmek istiyor.",
        "question": "Hangi sehir icin?",
        "options": {"istanbul": "Istanbul", "ankara": "Ankara"},
        "answerable": True, "correct_option": "ankara",
    },
    {
        "id": "tr-train-3", "language": "tr", "origin": "native", "source": "smoke", "split": "train",
        "state": "Durum: Kullanici hava durumunu ogrenmek istiyor ama konum belirtilmemis.",
        "question": "Hangi sehir icin?",
        "options": {"istanbul": "Istanbul", "ankara": "Ankara"},
        "answerable": False, "correct_option": None,
    },
    {
        "id": "tr-train-4", "language": "tr", "origin": "native", "source": "smoke", "split": "train",
        "state": "Durum: Kullanici config.yaml dosyasini incelemek istiyor.",
        "question": "Hangi arac kullanilmali?",
        "options": {"filesystem": "Yerel dosyalari oku", "web": "Internette ara"},
        "answerable": True, "correct_option": "filesystem",
    },
    {
        "id": "tr-val-1", "language": "tr", "origin": "native", "source": "smoke", "split": "validation",
        "state": "Durum: Kullanici config.yaml dosyasini incelemek istiyor.",
        "question": "Hangi arac kullanilmali?",
        "options": {"filesystem": "Yerel dosyalari oku", "web": "Internette ara"},
        "answerable": True, "correct_option": "filesystem",
    },
    {
        "id": "tr-val-2", "language": "tr", "origin": "native", "source": "smoke", "split": "validation",
        "state": "Durum: Kullanici hava durumunu ogrenmek istiyor ama konum belirtilmemis.",
        "question": "Hangi sehir icin?",
        "options": {"istanbul": "Istanbul", "ankara": "Ankara"},
        "answerable": False, "correct_option": None,
    },
]


def build_tiny_local_model(model_dir: Path) -> None:
    """Build and save a tiny GPT-2-style base model + tokenizer, no downloads."""
    from tokenizers.implementations import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

    model_dir.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(TRAIN_CORPUS, vocab_size=300, min_frequency=1, special_tokens=["<|endoftext|>"])
    bpe.save_model(str(model_dir))

    tokenizer = GPT2TokenizerFast(vocab_file=str(model_dir / "vocab.json"), merges_file=str(model_dir / "merges.txt"))
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.save_pretrained(str(model_dir))

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size, n_embd=8, n_layer=1, n_head=1, n_positions=64,
        attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
    )
    GPT2Model(config).save_pretrained(str(model_dir))


class TrainV1DecisionModelSmokeTest(unittest.TestCase):
    """One real subprocess CLI run, kept small enough for CPU/RAM-constrained runs."""

    @classmethod
    def setUpClass(cls):
        cls._workspace = tempfile.TemporaryDirectory()
        workspace = Path(cls._workspace.name)
        cls.model_dir = workspace / "model"
        cls.data_path = workspace / "data.jsonl"
        cls.output_dir = workspace / "out"

        build_tiny_local_model(cls.model_dir)
        cls.data_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in RECORDS), encoding="utf-8"
        )

        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--model-path", str(cls.model_dir),
                "--data", str(cls.data_path),
                "--output-dir", str(cls.output_dir),
                "--epochs", "1",
                "--batch-size", "2",
                "--max-length", "32",
                "--num-threads", "1",
            ],
            capture_output=True, text=True, timeout=180,
        )
        cls.process = result

    @classmethod
    def tearDownClass(cls):
        cls._workspace.cleanup()

    def test_cli_exits_successfully_with_a_json_result(self):
        self.assertEqual(self.process.returncode, 0, msg=self.process.stderr)
        payload = json.loads(self.process.stdout)
        self.assertIn("metadata", payload)
        self.assertIn("metrics", payload)
        self.assertTrue((self.output_dir / "checkpoints" / "best.pt").is_file())

    def test_metadata_records_reproducibility_fields(self):
        metadata = json.loads((self.output_dir / "train_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["seed"], 13)
        self.assertEqual(Path(metadata["local_model_source"]).resolve(), self.model_dir.resolve())
        self.assertEqual(metadata["hyperparameters"]["epochs"], 1)
        self.assertEqual(metadata["hyperparameters"]["num_threads"], 1)
        self.assertEqual(metadata["train_record_count"], 4)
        self.assertEqual(metadata["validation_record_count"], 2)
        self.assertEqual(metadata["selected_validation_checkpoint_epoch"], 0)
        self.assertGreater(metadata["calibration_temperature"], 0)
        self.assertGreaterEqual(metadata["answerability_threshold"], 0.0)
        self.assertLessEqual(metadata["answerability_threshold"], 1.0)

    def test_validation_metrics_file_has_expected_shape(self):
        metrics = json.loads((self.output_dir / "validation_metrics.json").read_text(encoding="utf-8"))
        for key in ("nll", "brier", "accuracy", "ece"):
            self.assertIn(key, metrics["decision"])
        for key in ("accuracy", "nll", "brier", "ece", "answerable_precision", "answerable_recall"):
            self.assertIn(key, metrics["answerability"])
        for key in ("coverage", "selective_accuracy"):
            self.assertIn(key, metrics["coverage"])

    def test_saved_checkpoint_files_exist(self):
        self.assertTrue((self.output_dir / "decision_heads.pt").is_file())
        self.assertTrue((self.output_dir / "backbone" / "config.json").is_file())
        self.assertTrue((self.output_dir / "backbone" / "tokenizer_config.json").is_file())
        self.assertTrue((self.output_dir / "training_data_fingerprints.json").is_file())

    def test_frozen_evaluation_cli_uses_saved_calibration(self):
        test_data = self.output_dir.parent / "frozen_test.jsonl"
        test_records = []
        for index, original in enumerate(RECORDS[-2:]):
            item = dict(original)
            item["id"] = f"test-{index}"
            item["split"] = "test"
            item["state"] = f"Bağımsız test durumu {index}"
            test_records.append(item)
        test_data.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in test_records), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_v1_decision_model.py"),
             "--model-dir", str(self.output_dir), "--data", str(test_data),
             "--output-dir", str(self.output_dir.parent / "test_result"),
             "--batch-size", "2", "--num-threads", "1"],
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        metrics = json.loads(result.stdout)
        metadata = json.loads((self.output_dir / "train_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["calibration_temperature"], metadata["calibration_temperature"])
        self.assertEqual(metrics["answerability_threshold"], metadata["answerability_threshold"])
        self.assertEqual(metrics["by_language"]["tr"]["record_count"], 2)
        self.assertEqual(len((self.output_dir.parent / "test_result" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_finalizes_saved_checkpoint_without_retraining(self):
        recovered = self.output_dir.parent / "recovered"
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--model-path", str(self.model_dir), "--data", str(self.data_path),
             "--output-dir", str(recovered), "--epochs", "1", "--batch-size", "2",
             "--max-length", "32", "--num-threads", "1",
             "--finalize-checkpoint", str(self.output_dir / "checkpoints" / "best.pt")],
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertNotIn("epoch 1/1", result.stderr)
        original = json.loads((self.output_dir / "validation_metrics.json").read_text(encoding="utf-8"))
        restored = json.loads((recovered / "validation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(original, restored)
        self.assertTrue((recovered / "backbone" / "config.json").is_file())

    def test_finalize_rejects_changed_training_data(self):
        changed_data = self.output_dir.parent / "changed_data.jsonl"
        changed_data.write_text(self.data_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--model-path", str(self.model_dir), "--data", str(changed_data),
             "--output-dir", str(self.output_dir.parent / "rejected"),
             "--epochs", "1", "--batch-size", "2", "--max-length", "32", "--num-threads", "1",
             "--finalize-checkpoint", str(self.output_dir / "checkpoints" / "best.pt")],
            capture_output=True, text=True, timeout=90,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint configuration or training data differs", result.stderr)
        self.assertFalse((self.output_dir.parent / "rejected" / "training_data_fingerprints.json").exists())

    def test_reloaded_checkpoint_scores_the_same_validation_request_deterministically(self):
        from transformers import AutoModel, AutoTokenizer

        _, validation_examples = TRAINER.load_examples(self.data_path)

        def load_fresh_model():
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir / "backbone", local_files_only=True)
            backbone = AutoModel.from_pretrained(self.output_dir / "backbone", local_files_only=True)
            model = TRAINER.DecisionModel(backbone)
            heads = torch.load(self.output_dir / "decision_heads.pt", map_location="cpu", weights_only=True)
            model.decision_head.load_state_dict(heads["decision_head"])
            model.answerability_head.load_state_dict(heads["answerability_head"])
            model.eval()
            return model, tokenizer

        model_a, tokenizer_a = load_fresh_model()
        scores_a, probabilities_a = TRAINER.predict(
            model_a, validation_examples, tokenizer_a, max_length=32, batch_size=2
        )
        model_b, tokenizer_b = load_fresh_model()
        scores_b, probabilities_b = TRAINER.predict(
            model_b, validation_examples, tokenizer_b, max_length=32, batch_size=2
        )

        self.assertEqual(scores_a, scores_b)
        self.assertEqual(probabilities_a, probabilities_b)
        for scores in scores_a:
            self.assertTrue(all(torch.isfinite(torch.tensor(list(scores.values())))))
        self.assertTrue(all(0.0 <= probability <= 1.0 for probability in probabilities_a))


if __name__ == "__main__":
    unittest.main()

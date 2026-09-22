"""Train the V1 CPU decision model: duplicated-branch scoring + answerability head.

This is V1's correctness-first trainer. Every option is encoded as an
independent branch (state + question + one option), matching the V0
duplicated-branch oracle in ``reflexmodels.causal_scoring``. No text is
generated; the model only ever produces bounded candidate scores and a
separate P(answerable) scalar. Packed shared-prefix execution is V2 work.

Two losses are trained jointly per record:
  - candidate decision cross-entropy, computed only for answerable records;
  - answerability binary cross-entropy, computed for every record.

Model selection, temperature calibration, and answerability-threshold
selection use split="validation" records only. Records with any other split
(including "test") are rejected before training starts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexmodels.answerability import answerability_metrics, select_answerability_threshold, selective_accuracy
from reflexmodels.causal_scoring import branch_prefix
from reflexmodels.scoring import softmax
from reflexmodels.serialization import QUESTION, STATE
from reflexmodels.split_policy import normalized_text_hash


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_module("validate_v1_decision_data", ROOT / "scripts" / "validate_v1_decision_data.py")


def answerability_text(state: str, question: str) -> str:
    """Encode state + question only, with no option, for the answerability head."""
    return f"{STATE}\n{state}\n{QUESTION}\n{question}"


@dataclass(frozen=True, slots=True)
class DecisionExample:
    """One V1 record reduced to what the trainer needs."""

    record_id: str
    split: str
    state: str
    question: str
    options: tuple[tuple[str, str], ...]
    answerable: bool
    correct_option: str | None


def load_examples(path: Path) -> tuple[list[DecisionExample], list[DecisionExample]]:
    """Load, validate, and split V1 records into train and validation examples.

    Reuses the shared V1 validator so this trainer can never silently accept
    malformed records or a frozen/test split under a different label.
    """
    records = VALIDATOR.load_records(path)
    VALIDATOR.validate(records)
    train_examples: list[DecisionExample] = []
    validation_examples: list[DecisionExample] = []
    for record in records:
        example = DecisionExample(
            record_id=record["id"],
            split=record["split"],
            state=record["state"],
            question=record["question"],
            options=tuple(record["options"].items()),
            answerable=record["answerable"],
            correct_option=record["correct_option"],
        )
        (train_examples if record["split"] == "train" else validation_examples).append(example)
    if not train_examples:
        raise ValueError("data file contains no split=train records")
    if not validation_examples:
        raise ValueError("data file contains no split=validation records")
    if not any(example.answerable for example in train_examples):
        raise ValueError("train split must contain answerable records for decision training")
    if {example.answerable for example in validation_examples} != {False, True}:
        raise ValueError("validation split must contain answerable and unanswerable records")
    return train_examples, validation_examples


def tokenize_and_pad(tokenizer: Any, texts: list[str], *, max_length: int):
    """Right-pad tokenized sequences, keeping the final ``max_length`` tokens.

    Truncation keeps the tail, not the head, of each sequence: for a decision
    branch the tail is the option text (and for the answerability input, the
    question), which is the part the model must attend to at its pooled
    final token. For inputs longer than ``max_length`` this trades away
    earlier ``state`` context to preserve that tail; choose ``max_length``
    generously if state text is long and information-dense.
    """
    import torch

    if not texts:
        raise ValueError("texts must be non-empty")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    encoded = []
    for text in texts:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise ValueError("tokenizer produced an empty sequence")
        encoded.append(ids[-max_length:])
    width = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    lengths = torch.zeros(len(encoded), dtype=torch.long)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        lengths[row] = len(ids)
    return input_ids, attention_mask, lengths


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    branch_input_ids: Any
    branch_attention_mask: Any
    branch_lengths: Any
    record_option_names: list[list[str]]
    record_option_ranges: list[tuple[int, int]]
    target_index: list[int]
    decision_mask: Any
    answerable_input_ids: Any
    answerable_attention_mask: Any
    answerable_lengths: Any
    answerable_labels: Any
    record_ids: list[str]


def make_batch(examples: list[DecisionExample], tokenizer: Any, *, max_length: int) -> TrainingBatch:
    """Build one minibatch of independent option branches plus answerability inputs.

    Options per record are variable-length, so branches from every record in
    the batch are flattened into one padded tensor; ``record_option_ranges``
    records each record's slice for later per-record softmax and masking.
    """
    import torch

    if not examples:
        raise ValueError("examples must be non-empty")
    branch_texts: list[str] = []
    record_option_names: list[list[str]] = []
    record_option_ranges: list[tuple[int, int]] = []
    target_index: list[int] = []
    for example in examples:
        start = len(branch_texts)
        prefix = branch_prefix(example.state, example.question)
        option_names = [name for name, _ in example.options]
        for name, text in example.options:
            branch_texts.append(prefix + text)
        record_option_names.append(option_names)
        record_option_ranges.append((start, len(branch_texts)))
        if example.answerable:
            if example.correct_option not in option_names:
                raise ValueError(f"record {example.record_id}: correct_option not among its options")
            target_index.append(option_names.index(example.correct_option))
        else:
            target_index.append(-1)
    branch_input_ids, branch_attention_mask, branch_lengths = tokenize_and_pad(
        tokenizer, branch_texts, max_length=max_length
    )
    answerable_texts = [answerability_text(example.state, example.question) for example in examples]
    answerable_input_ids, answerable_attention_mask, answerable_lengths = tokenize_and_pad(
        tokenizer, answerable_texts, max_length=max_length
    )
    decision_mask = torch.tensor([example.answerable for example in examples], dtype=torch.bool)
    answerable_labels = torch.tensor([float(example.answerable) for example in examples], dtype=torch.float32)
    return TrainingBatch(
        branch_input_ids=branch_input_ids,
        branch_attention_mask=branch_attention_mask,
        branch_lengths=branch_lengths,
        record_option_names=record_option_names,
        record_option_ranges=record_option_ranges,
        target_index=target_index,
        decision_mask=decision_mask,
        answerable_input_ids=answerable_input_ids,
        answerable_attention_mask=answerable_attention_mask,
        answerable_lengths=answerable_lengths,
        answerable_labels=answerable_labels,
        record_ids=[example.record_id for example in examples],
    )


class DecisionModel:
    """Wraps a local causal-LM backbone with two small scalar heads.

    No autoregressive generation happens anywhere in this class: the
    backbone's own LM head is never used, only its final hidden state.
    """

    def __init__(self, backbone: Any):
        import torch.nn as nn

        hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
        if not hidden_size:
            raise ValueError("backbone config must expose hidden_size or n_embd")
        self.backbone = backbone
        self.backbone_frozen = False
        self.decision_head = nn.Linear(hidden_size, 1)
        # Mean and max summaries make the option-set signal order invariant.
        # Without option features, identical state/question text with a
        # different candidate set would receive the same answerability score.
        self.answerability_head = nn.Linear(hidden_size * 3, 1)

    def parameters(self):
        import itertools

        return itertools.chain(
            self.backbone.parameters(), self.decision_head.parameters(), self.answerability_head.parameters()
        )

    def train(self) -> None:
        if self.backbone_frozen:
            self.backbone.eval()
        else:
            self.backbone.train()
        self.decision_head.train()
        self.answerability_head.train()

    def eval(self) -> None:
        self.backbone.eval()
        self.decision_head.eval()
        self.answerability_head.eval()

    def state_dict(self) -> dict[str, Any]:
        """Return live tensor references, not copies.

        Callers that need to keep a checkpoint across further training steps
        must persist it themselves (e.g. ``torch.save`` straight to a file)
        rather than holding this dict in RAM, since these are the same
        tensors the optimizer keeps mutating in place.
        """
        return {
            "decision_head": self.decision_head.state_dict(),
            "answerability_head": self.answerability_head.state_dict(),
            "backbone": self.backbone.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decision_head.load_state_dict(state["decision_head"])
        self.answerability_head.load_state_dict(state["answerability_head"])
        self.backbone.load_state_dict(state["backbone"])

    def _pooled_hidden(self, input_ids: Any, attention_mask: Any, lengths: Any):
        import torch

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        index = (lengths - 1).clamp(min=0)
        return hidden[torch.arange(hidden.size(0)), index]

    def score_branch_hidden(self, branch_hidden: Any):
        return self.decision_head(branch_hidden.to(self.decision_head.weight.dtype)).squeeze(-1)

    def decision_scores(self, input_ids: Any, attention_mask: Any, lengths: Any):
        return self.score_branch_hidden(self._pooled_hidden(input_ids, attention_mask, lengths))

    def answerability_logits(
        self, input_ids: Any, attention_mask: Any, lengths: Any,
        branch_hidden: Any, record_option_ranges: list[tuple[int, int]],
    ):
        import torch

        context_hidden = self._pooled_hidden(input_ids, attention_mask, lengths)
        option_mean = torch.stack([branch_hidden[start:end].mean(dim=0) for start, end in record_option_ranges])
        option_max = torch.stack([branch_hidden[start:end].max(dim=0).values for start, end in record_option_ranges])
        features = torch.cat((context_hidden, option_mean, option_max), dim=-1)
        return self.answerability_head(features.to(self.answerability_head.weight.dtype)).squeeze(-1)


def compute_losses(model: DecisionModel, batch: TrainingBatch, *, answerability_weight: float = 1.0):
    """Return (total_loss, decision_loss_value, answerability_loss_value, answerable_count).

    Decision cross-entropy is summed only over records whose ``decision_mask``
    is true; unanswerable records never contribute a decision gradient, only
    an answerability one.
    """
    import torch

    branch_hidden = model._pooled_hidden(batch.branch_input_ids, batch.branch_attention_mask, batch.branch_lengths)
    branch_scores = model.score_branch_hidden(branch_hidden)
    decision_losses = []
    for record_position, (start, end) in enumerate(batch.record_option_ranges):
        if not bool(batch.decision_mask[record_position]):
            continue
        log_probs = torch.log_softmax(branch_scores[start:end], dim=0)
        decision_losses.append(-log_probs[batch.target_index[record_position]])
    decision_loss = torch.stack(decision_losses).mean() if decision_losses else branch_scores.sum() * 0.0

    answerability_logits = model.answerability_logits(
        batch.answerable_input_ids, batch.answerable_attention_mask, batch.answerable_lengths,
        branch_hidden, batch.record_option_ranges,
    )
    answerability_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        answerability_logits, batch.answerable_labels
    )
    total = decision_loss + answerability_weight * answerability_loss
    return total, float(decision_loss.detach()), float(answerability_loss.detach()), len(decision_losses)


def predict(
    model: DecisionModel, examples: list[DecisionExample], tokenizer: Any, *, max_length: int, batch_size: int
) -> tuple[list[dict[str, float]], list[float]]:
    """Return per-record decision score maps and P(answerable), with no grad."""
    import torch

    model.eval()
    score_maps: list[dict[str, float]] = []
    answerable_probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = make_batch(chunk, tokenizer, max_length=max_length)
            branch_hidden = model._pooled_hidden(
                batch.branch_input_ids, batch.branch_attention_mask, batch.branch_lengths
            )
            branch_scores = model.score_branch_hidden(branch_hidden)
            for record_position, (range_start, range_end) in enumerate(batch.record_option_ranges):
                names = batch.record_option_names[record_position]
                values = branch_scores[range_start:range_end].tolist()
                score_maps.append(dict(zip(names, values)))
            logits = model.answerability_logits(
                batch.answerable_input_ids, batch.answerable_attention_mask, batch.answerable_lengths,
                branch_hidden, batch.record_option_ranges,
            )
            answerable_probabilities.extend(torch.sigmoid(logits).tolist())
    model.train()
    return score_maps, answerable_probabilities


def fit_decision_temperature(
    score_maps: list[dict[str, float]], correct_options: list[str], *, iterations: int = 80
) -> float:
    """Fit one positive scalar minimizing decision NLL on validation-only data.

    Mirrors ``reflexmodels.calibration.fit_temperature``'s golden-section
    search, but operates on variable-length per-record score maps because
    each V1 record can carry a different number of runtime-defined options.
    """
    if not score_maps or len(score_maps) != len(correct_options):
        raise ValueError("score_maps and correct_options must be non-empty and have equal length")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    def nll_at(temperature: float) -> float:
        total = 0.0
        for scores, correct in zip(score_maps, correct_options):
            probabilities = softmax({name: value / temperature for name, value in scores.items()})
            total += -math.log(max(probabilities[correct], 1e-12))
        return total / len(score_maps)

    lower, upper = math.log(0.05), math.log(10.0)
    ratio = (math.sqrt(5) - 1) / 2
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    for _ in range(iterations):
        if nll_at(math.exp(left)) <= nll_at(math.exp(right)):
            upper, right = right, left
            left = upper - ratio * (upper - lower)
        else:
            lower, left = left, right
            right = lower + ratio * (upper - lower)
    return math.exp((lower + upper) / 2)


def decision_validation_metrics(
    score_maps: list[dict[str, float]], correct_options: list[str], *, temperature: float, ece_bins: int = 15
) -> dict[str, float]:
    """Return NLL/Brier/accuracy/ECE for answerable validation records only."""
    if not score_maps or len(score_maps) != len(correct_options):
        raise ValueError("score_maps and correct_options must be non-empty and have equal length")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    nll_total = 0.0
    brier_total = 0.0
    pairs: list[tuple[float, bool]] = []
    for scores, correct in zip(score_maps, correct_options):
        probabilities = softmax({name: value / temperature for name, value in scores.items()})
        nll_total += -math.log(max(probabilities[correct], 1e-12))
        brier_total += sum((value - float(name == correct)) ** 2 for name, value in probabilities.items())
        predicted = max(probabilities, key=probabilities.__getitem__)
        pairs.append((probabilities[predicted], predicted == correct))
    count = len(score_maps)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(ece_bins)]
    for confidence, correct_flag in pairs:
        bins[min(int(confidence * ece_bins), ece_bins - 1)].append((confidence, correct_flag))
    ece = sum(
        len(bucket) / count
        * abs(sum(c for c, _ in bucket) / len(bucket) - sum(f for _, f in bucket) / len(bucket))
        for bucket in bins
        if bucket
    )
    return {
        "nll": nll_total / count,
        "brier": brier_total / count,
        "accuracy": sum(flag for _, flag in pairs) / count,
        "ece": ece,
    }


def select_calibration_and_threshold(
    validation_examples: list[DecisionExample],
    score_maps: list[dict[str, float]],
    answerable_probabilities: list[float],
    *,
    min_coverage: float = 0.0,
) -> dict[str, Any]:
    """Fit temperature and the answerability threshold on validation data only.

    Raises if any supplied example is not split="validation" so a training
    bug can never let train or frozen-evaluation data leak into calibration.
    """
    if len(validation_examples) != len(score_maps) or len(validation_examples) != len(answerable_probabilities):
        raise ValueError("validation_examples, score_maps, and answerable_probabilities must be aligned by position")
    if any(example.split != "validation" for example in validation_examples):
        raise ValueError("select_calibration_and_threshold must only be called with split=validation examples")
    answerable_indexes = [index for index, example in enumerate(validation_examples) if example.answerable]
    if not answerable_indexes:
        raise ValueError("validation split must contain at least one answerable record for calibration")
    answerable_score_maps = [score_maps[index] for index in answerable_indexes]
    correct_options = [validation_examples[index].correct_option for index in answerable_indexes]
    temperature = fit_decision_temperature(answerable_score_maps, correct_options)

    decision_correct: list[bool] = []
    for example, scores in zip(validation_examples, score_maps):
        if not example.answerable:
            decision_correct.append(False)
            continue
        predicted = max(scores, key=scores.__getitem__)
        decision_correct.append(predicted == example.correct_option)
    threshold_result = select_answerability_threshold(
        decision_correct, answerable_probabilities, min_coverage=min_coverage
    )
    return {"temperature": temperature, **threshold_result}


def run_training(
    *,
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    answerability_weight: float,
    min_coverage: float,
    num_threads: int | None,
    freeze_backbone: bool = False,
    finalize_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if not model_path.is_dir():
        raise ValueError("model-path must be an existing local checkpoint directory")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if answerability_weight < 0:
        raise ValueError("answerability_weight must be non-negative")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between zero and one")

    import torch
    from transformers import AutoModel, AutoTokenizer

    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)

    random.seed(seed)
    torch.manual_seed(seed)

    train_examples, validation_examples = load_examples(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # AutoModel (not AutoModelForCausalLM): this trainer only ever consumes
    # hidden states, so loading the LM head would waste CPU/RAM computing
    # vocabulary logits that are never used.
    backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
    model = DecisionModel(backbone)
    if freeze_backbone:
        model.backbone.requires_grad_(False)
        model.backbone_frozen = True
    output_dir.mkdir(parents=True, exist_ok=True)
    training_fingerprints = {
        "normalization": "strip+casefold+sha256 of state text",
        "train_state_hashes": sorted({normalized_text_hash(example.state) for example in train_examples}),
        "validation_state_hashes": sorted({normalized_text_hash(example.state) for example in validation_examples}),
    }
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_epoch = -1
    best_score = math.inf
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    run_config = {
        "model_path": str(model_path.resolve()), "data_sha256": data_sha256,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "max_length": max_length, "seed": seed,
        "answerability_weight": answerability_weight, "min_coverage": min_coverage,
        "freeze_backbone": freeze_backbone,
    }
    if finalize_checkpoint is None:
        (output_dir / "training_data_fingerprints.json").write_text(
            json.dumps(training_fingerprints, indent=2), encoding="utf-8"
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate
        )
        model.train()
        order = list(range(len(train_examples)))
        for epoch in range(epochs):
            random.shuffle(order)
            train_loss_total = 0.0
            train_batches = 0
            for start in range(0, len(order), batch_size):
                chunk = [train_examples[index] for index in order[start : start + batch_size]]
                batch = make_batch(chunk, tokenizer, max_length=max_length)
                loss, _, _, _ = compute_losses(model, batch, answerability_weight=answerability_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_total += float(loss.detach())
                train_batches += 1

            validation_score_maps, validation_answerable_probs = predict(
                model, validation_examples, tokenizer, max_length=max_length, batch_size=batch_size
            )
            answerable_indexes = [index for index, example in enumerate(validation_examples) if example.answerable]
            raw_metrics = decision_validation_metrics(
                [validation_score_maps[index] for index in answerable_indexes],
                [validation_examples[index].correct_option for index in answerable_indexes],
                temperature=1.0,
            )
            answerability_loss = torch.nn.functional.binary_cross_entropy(
                torch.tensor(validation_answerable_probs, dtype=torch.float32).clamp(1e-6, 1 - 1e-6),
                torch.tensor([float(example.answerable) for example in validation_examples], dtype=torch.float32),
            ).item()
            selection_score = raw_metrics["nll"] + answerability_loss
            improved = selection_score < best_score
            if improved:
                best_score = selection_score
                best_epoch = epoch
                temporary_path = checkpoint_dir / "best.tmp"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch + 1, "validation_score": best_score,
                            "run_config": run_config}, temporary_path)
                os.replace(temporary_path, best_checkpoint_path)
            print(
                f"epoch {epoch + 1}/{epochs}: train_loss={train_loss_total / max(train_batches, 1):.4f} "
                f"val_decision_nll={raw_metrics['nll']:.4f} val_decision_accuracy={raw_metrics['accuracy']:.4f} "
                f"val_answerability_bce={answerability_loss:.4f}"
                f"{' (new best; checkpoint saved)' if improved else ''}",
                file=sys.stderr,
                flush=True,
            )
    else:
        saved_state = torch.load(finalize_checkpoint, map_location="cpu", weights_only=True)
        if saved_state.get("run_config") != run_config:
            raise ValueError("checkpoint configuration or training data differs from supplied arguments")
        best_epoch = int(saved_state["epoch"]) - 1
        best_score = float(saved_state["validation_score"])
        best_checkpoint_path = finalize_checkpoint
        (output_dir / "training_data_fingerprints.json").write_text(
            json.dumps(training_fingerprints, indent=2), encoding="utf-8"
        )

    if best_epoch < 0:
        raise RuntimeError("no validation checkpoint was selected")
    saved_state = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved_state["model"])

    validation_score_maps, validation_answerable_probs = predict(
        model, validation_examples, tokenizer, max_length=max_length, batch_size=batch_size
    )
    calibration = select_calibration_and_threshold(
        validation_examples, validation_score_maps, validation_answerable_probs, min_coverage=min_coverage
    )
    answerable_indexes = [index for index, example in enumerate(validation_examples) if example.answerable]
    decision_metrics = decision_validation_metrics(
        [validation_score_maps[index] for index in answerable_indexes],
        [validation_examples[index].correct_option for index in answerable_indexes],
        temperature=calibration["temperature"],
    )
    answerability_evaluation = answerability_metrics(
        [int(example.answerable) for example in validation_examples], validation_answerable_probs
    )
    decision_correct = []
    for example, scores in zip(validation_examples, validation_score_maps):
        if not example.answerable:
            decision_correct.append(False)
            continue
        decision_correct.append(max(scores, key=scores.__getitem__) == example.correct_option)
    coverage = selective_accuracy(decision_correct, validation_answerable_probs, threshold=calibration["threshold"])

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decision_head": model.decision_head.state_dict(),
            "answerability_head": model.answerability_head.state_dict(),
        },
        output_dir / "decision_heads.pt",
    )
    # Persist a self-contained checkpoint in both training modes. In full
    # fine-tuning the backbone weights change; in the frozen pilot this also
    # protects the exact base revision used by the heads.
    model.backbone.save_pretrained(output_dir / "backbone")
    tokenizer.save_pretrained(output_dir / "backbone")

    metadata = {
        "seed": seed,
        "local_model_source": str(model_path),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "answerability_weight": answerability_weight,
            "min_coverage": min_coverage,
            "num_threads": torch.get_num_threads(),
            "freeze_backbone": freeze_backbone,
        },
        "train_record_count": len(train_examples),
        "validation_record_count": len(validation_examples),
        "selected_validation_checkpoint_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path),
        "calibration_temperature": calibration["temperature"],
        "answerability_threshold": calibration["threshold"],
    }
    metrics = {
        "decision": decision_metrics,
        "answerability": answerability_evaluation,
        "coverage": coverage,
    }
    (output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"metadata": metadata, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="Local causal-LM checkpoint directory")
    parser.add_argument("--data", required=True, type=Path, help="V1 JSONL file with train and validation records")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for checkpoints and metadata")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--answerability-weight", type=float, default=1.0)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, help="CPU threads for PyTorch")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the decision heads for a low-cost pilot")
    parser.add_argument("--finalize-checkpoint", type=Path,
                        help="Skip training and finalize an existing best.pt using the exact original arguments")
    args = parser.parse_args()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise RuntimeError("install compatible torch and transformers in .venv before running") from error

    result = run_training(
        model_path=args.model_path,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        answerability_weight=args.answerability_weight,
        min_coverage=args.min_coverage,
        num_threads=args.num_threads,
        freeze_backbone=args.freeze_backbone,
        finalize_checkpoint=args.finalize_checkpoint,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

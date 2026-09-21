"""Fine-tune and evaluate one CPU fixed-label encoder baseline from a JSON config.

Requires the project virtual environment with compatible ``torch`` and
``transformers`` packages. This runner intentionally has no GPU mode: CPU
latency is a primary project metric.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reflexmodels.calibration import fit_temperature, temperature_scale
from reflexmodels.cpu_benchmark import measure_callable
from reflexmodels.metrics import classification_metrics
from reflexmodels.split_policy import apply_overlap_policy


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not isinstance(item.get("label"), int):
            raise ValueError(f"{path}:{number}: expected text and integer label")
        records.append(item)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def select_run(config: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [run for run in config.get("runs", []) if run.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"config must contain exactly one run named {name!r}")
    return matches[0]


def split_filename(run: dict[str, Any], split: str) -> str:
    """Return the configured filename, accepting ``validation_split`` for dev."""
    key = "validation_split" if split == "dev" else f"{split}_split"
    value = run.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run is missing a non-empty {key}")
    return value


def label_count(splits: dict[str, list[dict[str, Any]]]) -> int:
    labels = {record["label"] for records in splits.values() for record in records}
    if not labels or labels != set(range(max(labels) + 1)):
        raise ValueError("labels must be contiguous zero-based IDs across splits")
    return len(labels)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def probabilities(logits: list[list[float]], temperature: float = 1.0) -> list[list[float]]:
    import math

    result = []
    for row in logits:
        scaled = [value / temperature for value in row]
        offset = max(scaled)
        weights = [math.exp(value - offset) for value in scaled]
        total = sum(weights)
        result.append([value / total for value in weights])
    return result


def predict(model: Any, tokenizer: Any, records: list[dict[str, Any]], *, batch_size: int, max_length: int, torch: Any) -> list[list[float]]:
    model.eval()
    output: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            encoded = tokenizer([record["text"] for record in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            logits = model(**encoded).logits.tolist()
            output.extend(logits)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, help="Local checkpoint directory; avoids an implicit network download")
    parser.add_argument("--overlap-policy", choices=["report_only", "drop_train_dev_overlapping_test"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-threads", type=int, help="CPU threads for PyTorch; default keeps its configured value")
    args = parser.parse_args()
    numerical = [args.epochs, args.batch_size, args.max_length]
    if args.num_threads is not None:
        numerical.append(args.num_threads)
    if min(numerical) < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, max length, and optional thread count must be positive")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install compatible torch and transformers in .venv before running") from error

    config = json.loads(args.config.read_text(encoding="utf-8"))
    run = select_run(config, args.run)
    dataset_dir = ROOT / config["dataset"]["prepared_directory"]
    supplied = {name: read_jsonl(dataset_dir / split_filename(run, name)) for name in ("train", "dev", "test")}
    expected_language = run["language"]
    if any(record.get("language") != expected_language for records in supplied.values() for record in records):
        raise ValueError("run language does not match every selected prepared record")
    splits, removed = apply_overlap_policy(supplied, policy=args.overlap_policy)
    classes = label_count(splits)
    seed = int(config["seed"])
    set_seed(seed, torch)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    num_threads = torch.get_num_threads()
    model_source = str(args.model_path) if args.model_path else run["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=classes, ignore_mismatched_sizes=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state, best_nll = None, float("inf")
    train = splits["train"]
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}: training {len(train)} records", flush=True)
        model.train()
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = [train[index] for index in order[start : start + args.batch_size]]
            encoded = tokenizer([record["text"] for record in batch], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            labels = torch.tensor([record["label"] for record in batch], dtype=torch.long)
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        dev_logits = predict(model, tokenizer, splits["dev"], batch_size=args.batch_size, max_length=args.max_length, torch=torch)
        dev_probs = probabilities(dev_logits)
        dev_nll = classification_metrics([r["label"] for r in splits["dev"]], dev_probs)["nll"]
        print(f"epoch {epoch + 1}/{args.epochs}: validation_nll={dev_nll:.6f}", flush=True)
        if dev_nll < best_nll:
            best_nll = dev_nll
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no model state was selected")
    model.load_state_dict(best_state)
    dev_logits = predict(model, tokenizer, splits["dev"], batch_size=args.batch_size, max_length=args.max_length, torch=torch)
    temperature = fit_temperature([r["label"] for r in splits["dev"]], probabilities(dev_logits))
    test_logits = predict(model, tokenizer, splits["test"], batch_size=args.batch_size, max_length=args.max_length, torch=torch)
    raw_test = probabilities(test_logits)
    calibrated_test = [temperature_scale(row, temperature) for row in raw_test]
    one = splits["test"][:1]
    cpu = measure_callable(lambda: predict(model, tokenizer, one, batch_size=1, max_length=args.max_length, torch=torch), warmup=int(config["cpu_benchmark"]["warmup"]), repetitions=int(config["cpu_benchmark"]["repetitions"]))
    labels = [record["label"] for record in splits["test"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "test_predictions.jsonl"
    predictions_path.write_text("".join(json.dumps({"id": record["id"], "language": record["language"], "label": record["label"], "probabilities": row}, ensure_ascii=False) + "\n" for record, row in zip(splits["test"], calibrated_test)), encoding="utf-8")
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "run": run["name"], "model": run["model"], "model_source": model_source, "language": expected_language, "seed": seed,
        "dataset": config["dataset"], "overlap_policy": args.overlap_policy, "removed_counts": removed,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_length": args.max_length, "num_threads": num_threads},
        "validation_best_nll": best_nll, "temperature": temperature,
        "test_metrics_raw": classification_metrics(labels, raw_test),
        "test_metrics_temperature_calibrated": classification_metrics(labels, calibrated_test),
        "cpu_single_request": cpu, "hardware": platform.platform(), "torch": torch.__version__,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

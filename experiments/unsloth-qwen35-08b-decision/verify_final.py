"""Evaluate the saved adapter independently, without refitting calibration."""
import os, json, math, sys
from pathlib import Path
os.environ["HF_HOME"] = "/workspace/reflex-decision/cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
from unsloth import FastDecisionModel
from datasets import load_dataset
run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "manifest.json").read_text())
model, tokenizer = FastDecisionModel.from_pretrained(str(run_dir / "adapter"), load_in_4bit=True, local_files_only=True)
test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test", revision=manifest["dataset_revision"])
items, report = FastDecisionModel.build_dataset(test, tokenizer, model)
metrics = FastDecisionModel.evaluate(model, tokenizer, items)
expected = json.loads((run_dir / "test-calibrated.json").read_text())
result = {"metrics": metrics, "build_report": report, "accuracy_delta_from_presave": metrics["accuracy"] - expected["accuracy"], "calibration_refitted": False}
(run_dir / "test-reloaded.json").write_text(json.dumps(result, indent=2))
print(result)
assert math.isfinite(metrics["loss"])
assert abs(result["accuracy_delta_from_presave"]) <= 0.005, "Saved adapter test accuracy differs by more than 0.5 percentage points"

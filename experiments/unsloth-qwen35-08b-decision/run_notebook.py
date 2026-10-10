"""Execute the prepared notebook Python cells; dependencies installed separately."""
import json, os, sys
from pathlib import Path
secrets = json.loads(Path("/workspace/reflex-decision/.credentials.json").read_text())
os.environ["HF_TOKEN"] = secrets["HF_API_KEY"]
os.environ["WANDB_API_KEY"] = secrets["WANDB_API_KEY"]
os.environ["HF_HOME"] = "/workspace/reflex-decision/cache/huggingface"
os.environ["WANDB_DIR"] = "/workspace/reflex-decision"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Avoid version-dependent Torch compile overhead while preserving the recipe.
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
mode = sys.argv[1]
assert mode in ("smoke", "overfit", "overfit300", "sft")
os.environ["DECISION_SMOKE"] = "0" if mode == "sft" else "1"
namespace = {"__name__": "__main__", "mode": mode}
notebook = json.loads(Path("Qwen3_5_08B-Decision.ipynb").read_text())
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code": continue
    source = "".join(cell["source"])
    if source.startswith("%%"): continue
    if mode.startswith("overfit"):
        source = source.replace('Path("runs/smoke" if SMOKE else "runs/notebook-60steps")', 'Path("runs/" + mode)')
        source = source.replace("train_items = train_items[:32]", "train_items = train_items[:32]\n    eval_items = train_items[:]")
        source = source.replace("max_steps = 2 if SMOKE else 60", "max_steps = 300 if SMOKE else 60" if mode == "overfit300" else "max_steps = 60 if SMOKE else 60")
        source = source.replace("weight_decay = 0.01", "weight_decay = 0.0")
        source = source.replace("eval_items = eval_items[:16]", "eval_items = train_items[:]")
        source = source.replace("learning_rate = 2e-4", "learning_rate = 5e-4")
        source = source.replace('lr_scheduler_type = "cosine"', 'lr_scheduler_type = "constant"')
        source = source.replace("trainer = DecisionTrainer(", "trainer = DecisionTrainer(\n    head_learning_rate = 1e-3,")
        source = source.replace("gradient_accumulation_steps = 4", "gradient_accumulation_steps = 1")
    print(f"Executing notebook cell {index}, mode={mode}", flush=True)
    exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
# Verify adapter reload and decision inference in every mode.
from unsloth import FastDecisionModel
import torch, gc
run_dir = namespace["RUN_DIR"]
namespace.pop("trainer", None)
namespace.pop("model", None)
gc.collect(); torch.cuda.empty_cache()
model, tokenizer = FastDecisionModel.from_pretrained(str(run_dir / "adapter"), load_in_4bit=True, local_files_only=True)
FastDecisionModel.for_inference(model)
answer = FastDecisionModel.predict(model, tokenizer, "Please refund my duplicate payment.", {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})
namespace["record_metric"]("reload-prediction", answer)
if mode == "sft":
    model.save_pretrained_merged(str(run_dir / "merged"))
    import wandb
    if wandb.run:
        namespace["record_metric"]("wandb", {"id": wandb.run.id, "url": wandb.run.url, "name": wandb.run.name})
        wandb.finish()
(run_dir / "complete.json").write_text(json.dumps({"mode": mode, "completed": True}))

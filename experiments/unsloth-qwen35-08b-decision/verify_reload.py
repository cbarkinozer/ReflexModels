import os, json, sys
from pathlib import Path
os.environ["HF_HOME"] = "/workspace/reflex-decision/cache/huggingface"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
from unsloth import FastDecisionModel
run_dir = Path(sys.argv[1])
model, tokenizer = FastDecisionModel.from_pretrained(str(run_dir / "adapter"), load_in_4bit=True, local_files_only=True)
FastDecisionModel.for_inference(model)
result = FastDecisionModel.predict(model, tokenizer, "Please refund my duplicate payment.", {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})
(run_dir / "reload-prediction.json").write_text(json.dumps(result, indent=2))
(run_dir / "complete.json").write_text(json.dumps({"mode": "smoke", "completed": True, "reload": "separate process; local cache"}))
print(result)

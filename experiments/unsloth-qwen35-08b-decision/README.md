# Qwen3.5-0.8B Unsloth decision notebook reproduction

User-authorized independent reproduction of the supplied Unsloth notebook. This experiment does not advance or replace Reflex V1/V2 and does not use frozen phone or ReflexBench examples for training. Mac training and Android integration follow later.

## Recipe

Qwen3.5-0.8B, typed-decisions/all, 4-bit LoRA r=16/alpha=16, seed 3407, context 2048, batch 8, accumulation 4, 60 steps, warmup 10, backbone LR 2e-4, cosine schedule, weight decay 0.01. Remaining defaults come from Unsloth; capture their resolved values on the pod before the full run. This is the notebook recipe adapted to 0.8B, not an exact reproduction of the blog's mixed-source benchmark. Notebook installation commands are preserved from the supplied source and require remote validation; do not reuse the old LLM Bender September dependency pins.

## Pod

One RTX 4090 24 GB, Linux x86_64, working NVIDIA driver, SSH access, at least 50 GB disk (80 GB preferred). Verify the persistent mount. Keep the Python environment, downloads, caches, logs and checkpoints there. Use a fresh non-interactive SSH session to verify cache environment variables persist. Read LLM Bender Lab's environment-setup-gotchas before setup. No API credentials are required for the public source data/model.

## Execution gates

1. Validate driver, CUDA Torch, Unsloth imports and dependency compatibility before model download.
2. Execute the notebook with DECISION_SMOKE=1 in the kernel environment. This uses 32 training items, 16 holdout/test items and two optimizer steps. Verify finite loss, evaluation, calibration and artifact saving. Smoke scores are not research results.
3. Reload the adapter and run the example decision request; confirm saving/loading works. If learning behavior is suspect, perform a separate tiny-overfit diagnosis before proceeding.
4. Execute the normal notebook in a fresh kernel without DECISION_SMOKE. It preserves batch 8/accumulation 4 and 60 steps. Runs refuse to overwrite directories; archive a failed attempt before retrying.
5. Retain the executed notebook with outputs. Record resolved TrainingArguments/head LR, actual dataset sizes, skip reports, Unsloth commits/versions, script SHA256, split overlap counts and GPU peak memory in the run artifacts. The prepared notebook already captures model/dataset revisions, package versions, GPU details and before/after/calibrated metrics. Check train/calibration/test overlap before interpreting scores; disclose any changes to source splits.
6. Evaluate the independent typed-decisions test and report its actual item count. The before score is a fresh decision head, not prompted Qwen. Do not compare it numerically to the phone's 50-case routing accuracy as if the tasks match.
7. Enable the notebook's merged-save cell after training. GGUF export and desktop/native Android inference parity require a separate validated export step; the supplied notebook does not perform GGUF export.

## Status

Completed training, calibration, independent test evaluation and saved-adapter reload on 2026-10-10. Test accuracy: 34.4% before, 65.7% after, 65.75% after reload. Local backup verified all 77 files. See [REPORT.md](REPORT.md) for results, diagnostic limitations and remaining work. W&B actual training was initially recorded offline; the replacement key worked and the original run is now synced to https://wandb.ai/c-barkinozer/reflexmodels/runs/4khq3xdg. Android and Mac mini experiments remain separate later steps.

## Tracking and credentials

Actual fine-tuning only: W&B project `reflexmodels`, display name `qwen35-08b-decision-typed-decisions-lora16-lr2e4-60steps`. Smoke and tiny-overfit runs use `WANDB_MODE=disabled` and `report_to=none`. User authorized using LLM Bender Lab HF/W&B credentials; those are transferred privately to a mode-600 pod file and never committed or included in artifacts. run_notebook.py executes the notebook cells outside Jupyter and verifies adapter reload. setup_pod.sh creates an isolated environment; final resolved versions are authoritative.

## Structured inference

`serve_decisions.py` exposes our trained model through a minimal localhost HTTP adapter at `/v1/systemone`, using `FastDecisionModel.predict`. This is a demonstration adapter, not the Unsloth Studio server. `decision-request.json` exercises choice, noul and score together. It returns the predictor's structured fields and probabilities; no text is generated. Current validation uses CUDA on the pod. Native Android, GGUF inference and Mac CPU/unified-memory support for our custom model remain unvalidated.

On the prepared pod:

```bash
cd /workspace/reflex-decision
.venv/bin/python serve_decisions.py
```

From another shell on the pod:

```bash
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' --data-binary @decision-request.json
```

For Unsloth Studio's official API path, the training documentation specifies `UNSLOTH_SYSTEMONE_MODEL=/path/to/merged unsloth studio ...`. The separate Laya serving guide's default downloads, CPU speed and option-trimming behavior must not be assumed to describe our Qwen model. The model predictor uses the Clef path; serving can read beyond the training context.

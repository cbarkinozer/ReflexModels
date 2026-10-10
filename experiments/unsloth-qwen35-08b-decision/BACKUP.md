# Completed local backup

All required training and inference artifacts are downloaded and hash-verified. The GPU pod can be closed without losing this run. See `backup-status.json` for final verification counts and archive hashes.

Local root: `experiments/unsloth-qwen35-08b-decision/runs/pod-backup/`.

- `merged/`: full merged model, decision head, calibration/configs and tokenizer; 13 files verified against pod hashes.
- `checkpoints/`: final training checkpoint, including optimizer/scheduler/RNG/trainer state; 15 files verified.
- `extracted/runs/notebook-60steps/adapter/`: original trained LoRA adapter and decision head.
- `extracted/runs/notebook-60steps/`: original before/after/calibration/test/reload metrics, complete resolved TrainingArguments, head recipe, manifest, GPU information and package freeze.
- `extracted/runs/balanced-benchmarks/`: gold-label scores, exact selected rows, full schemas, source revisions and selection seed.
- `extracted/runs/data-snapshot/`: original typed-decisions training and test data.
- `extracted/runs/inference-demo/`: HTTP request/response records, schema checks and merged-model reload result.
- `extracted/`: actual executed source scripts/notebook, setup/failure/smoke/diagnostic/training/evaluation/service logs and inventories.
- `extracted/wandb/offline-run-20261010_165410-4khq3xdg/`: offline actual-fine-tuning run, ID `4khq3xdg`, project `reflexmodels`.
- `artifacts-portable.tar` and `artifacts-supplement.tar`: compact original and supplementary archives. The original archive's one diagnostic-log symlink was materialized in the extracted backup and verified.

Weights/raw runs are Git-ignored; the report, source, result registry and completion status remain outside that ignored directory. Credentials are excluded. Python environment/cache binaries and redundant diagnostic model weights are not needed to preserve this run; exact versions, sources and diagnostic configs/logs are retained. Recreate the environment from `setup_pod.sh` and the authoritative captured package freeze. Restore data/model paths before executing the pod-oriented scripts on another machine.

The demonstration HTTP service currently uses CUDA and the trained adapter, not native Mac/Android inference. Its source and request example are preserved, so leaving the service running is not necessary. W&B cloud sync completed using the replacement key: https://wandb.ai/c-barkinozer/reflexmodels/runs/4khq3xdg. The earlier key failed authentication.

reflex-1 GGUF exports are downloaded: `runs/pod-backup/reflex-1-gguf/`, including Q8_0, Q4_K_M, optional vision projector, export metadata, SHA-256 inventory, CPU API responses and server logs. All four exported artifact hashes match the pod. Converter source archive hash verified; exact llama.cpp revision, environment freeze, build/export/failure/verification logs are preserved in pod-backup. W&B sync log is backed up. No GPU dependency remains for this stage; the pod can be closed.

Cost snapshot and calculation are preserved in [COST.md](COST.md) and cost.json. At 2h 7m × $0.363/hour, estimated base cost is $0.76835, excluding internet and any later runtime. Final billing/shutdown remain unconfirmed. Future closeout requirements are in ../../EXPERIMENT_CLOSEOUT.md. reflex-1 remains unpublished at the user’s request.

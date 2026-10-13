# reflex-1-beta

**Renamed 2026-10-14:** this model was never published, so the user renamed it `reflex-1-beta` and the folder was physically moved from `releases/reflex-1/` to `releases/reflex-1-beta/` once the combined phone+general model took the `reflex-1` name instead. Cross-references inside this file to "reflex-1" below are left as historical text describing this model under its original name. The current model is [reflex-1](../reflex-1/README.md).

First ReflexModels Qwen3.5-0.8B decision-head model, trained on public typed-decisions using Unsloth. Research release prepared locally; not published to a model hub yet.

Supports structured `choice`, `noul` (yes/no probability), and `score` questions. It scores options without generating text. Tested with FastDecisionModel and our local `/v1/systemone` HTTP adapter. The adapter accepts `reflex-1`; it is not an official Unsloth Studio deployment.

## Measured results

| Gold-label test | Accuracy |
| --- | --- |
| typed-decisions: 400 records / 2,000 decisions | 65.6–65.75% |
| BANKING77: 462 balanced examples, 77 options | 44.59% |
| CLINC150: 450 balanced examples, 150 options | 47.11% |
| Training holdout: 80 records / 400 decisions | 66.75% |

These results do not reproduce the blog's mixed-source benchmark. Training used typed-decisions only, LoRA rank 16, 60 steps, seed 3407. Calibration was fitted on the training holdout, never the test sets. BANKING77 / CLINC150 calibration error was high (0.280 / 0.340); validate thresholds for your application.

## Local artifacts and reproduction

Canonical merged checkpoint: `../../experiments/unsloth-qwen35-08b-decision/runs/pod-backup/merged/`.
Original adapter, optimizer checkpoint, exact configs, dataset selections, logs and source revisions are preserved in the same pod-backup folder. See the experiment's REPORT.md and BACKUP.md for provenance and restore instructions.

Q8_0 (803 MiB) and Q4_K_M (520 MiB) GGUF exports completed and each passed a CPU `/v1/systemone` request with zero output tokens. Files are in `../../experiments/unsloth-qwen35-08b-decision/runs/pod-backup/reflex-1-gguf/`; the optional vision projector is preserved too. GGUF must retain the Clef decision head and temperatures; an ordinary text-only converter is insufficient. Android A32 loading, probability parity, memory, latency and task accuracy are not yet validated. Both GGUF tests flipped a close refund decision relative to the merged PyTorch 4-bit reload: yes probability 0.5015 became 0.4941 (Q8) / 0.4804 (Q4). Billing and urgency choices agreed. Full benchmark parity is not established. CPU single-request times on the pod were 1.54 / 1.32 seconds; these are not phone benchmarks.

The Qwen base model and typed-decisions dataset declare Apache-2.0. BASE_MODEL_LICENSE preserves the upstream license; DATASET_CARD.md preserves the training-revision dataset card. The weights were modified by decision-head and LoRA training on 2026-10-10. Unsloth/notebook source retains its own license. No public upload has been performed.

## Run the GGUF Decision API

Use a decision-capable llama.cpp build. The verified source revision is `aa94f2086147cd4bfbcff818ce998dffce5fcb39`, preserved as `llama-decision-source.tar.gz` in pod-backup. Build `llama-server`, then:

```bash
llama-server -m /path/to/model-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 4096 -ngl 0
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' --data-binary @decision-request.json
```

Use the experiment's decision-request.json, optionally setting model to reflex-1. The tested llama.cpp server reports the loaded file path in its response. For text decisions the vision projector is not needed. Android requires an ARM64 build and an appropriate phone context budget. Do not assume that server `confidence` is interchangeable with the Python predictor; threshold the selected option probability and validate it for your task.

Publication is deferred at the user’s request. Cost notes: ../../experiments/unsloth-qwen35-08b-decision/COST.md.

# reflex-1

**Renamed 2026-10-14:** this is the second ReflexModels decision-head model overall, but since the original (single-domain, typed-decisions-only) model was never published, it was renamed `reflex-1-beta` and this combined model takes the `reflex-1` name instead.

Built with Unsloth's `FastDecisionModel`/`DecisionTrainer` (the "Clef" architecture: LoRA-adapted causal backbone plus a trainable classification head for `choice`/`noul`/`score` questions). Supersedes the unpublished single-domain [reflex-1-beta](../reflex-1-beta/README.md). Research release prepared locally; not published to a model hub.

Trained on a **combined phone-routing + general mixture** (781 phone examples + 781 class-preserving general examples: 391 typed-decisions, 195 BANKING77, 195 CLINC150; equal domain weights), independently from base `unsloth/Qwen3.5-0.8B`, LoRA rank/alpha 16, LR `2e-4`, seed 3407. Two independent combined runs were trained (80 steps and 120 steps); the **120-step run's checkpoint-100** was selected on validation and is the model exported here.

## Measured results — validation only, no held-out test yet

| Checkpoint | Phone val | General val | Equal-domain mean |
| --- | ---: | ---: | ---: |
| Combined 80-step (best) | 98.89% | 74.49% | 86.67% |
| Combined 120-step, step 80 | 98.89% | 77.35% | 88.12% |
| **Combined 120-step, step 100 (selected)** | **98.89%** | **78.39%** | **88.64%** |
| Combined 120-step, step 120 | 98.89% | 77.80% | 88.34% |

Calibration fitted and OK (`calibration_ok: true`); post-calibration ECE `.0377` vs pre-calibration `.0511`. These are **validation-set** numbers computed automatically during training, not an independent test. The frozen held-out test protocol (historical phone, typed-decisions official test, BANKING77, CLINC150 — same four sets reflex-1-beta's README reports real numbers on) is defined in [`../../experiments/decision-followups/HELDOUT_EVALUATION_20261013.md`](../../experiments/decision-followups/HELDOUT_EVALUATION_20261013.md) but **has not been run** for this model yet. Treat reflex-1 as validation-qualified, not test-qualified, until that runs.

Known leakage/interpretation caveats (phone validation reuse across pilots, lexical overlap, in-domain-not-zero-shot general sets) are documented in the same held-out-evaluation doc — read it before quoting the phone 98.89% as a generalization claim.

## Local artifacts and reproduction

Training ran on a rented Vast.ai RTX 4090 (instance `55798314`, now destroyed). Full provenance — exact commands, environment freeze, data/split hashes, leakage audit, cost — is in [`../../experiments/decision-followups/VAST_FINE_TUNE_HANDOFF_20261013.md`](../../experiments/decision-followups/VAST_FINE_TUNE_HANDOFF_20261013.md). Total pod runtime was approximately 2.0 h (training + merge + GGUF export); provisional cost at `$0.443/h` plus network ≈ **$0.90–0.95** (last confirmed snapshot before shutdown: 1.66 h / ≈$0.79, plus the ~20–25 min GGUF export step not separately billed-and-captured — treat as an estimate, not a final invoice).

Backed up locally (SHA-256 verified against the pod before it was destroyed):
- Full project workspace (all three training runs' checkpoints/adapters/logs/manifests, prepared data, smoke-test runs): `pocket-agents-lab/runs/decision-followups-vast-20261013/reflex-decision-artifacts-20261013.tar.gz` (4.6 GB), sha256 `2f4fd5a651880e8504013ac525f6f204f63690de5e04cc42702d397cda0f73be`.
- Base-model Hugging Face cache: `pocket-agents-lab/runs/decision-followups-vast-20261013/hf-hub-cache-20261013.tar.gz` (1.3 GB), sha256 `e586d4ef1737edd9899c3e137a4e848365fa31b51ae6ecba483d7f1c21568d81`.
- Selected adapter path inside the archive: `runs/combined-balanced-scratch-120step-s3407/adapter` (best-checkpoint reload of step 100).

## GGUF export

Merged with base (`FastDecisionModel.save_pretrained_merged`) and exported via `unsloth.models.decision_gguf.export_decision_gguf` in the same "clef" layout as reflex-1-beta — a text-model GGUF plus a vision-projector sidecar GGUF (the sidecar is untouched by this LoRA fine-tune, hence byte-identical to reflex-1-beta's). One snag: Unsloth's auto-fetch for the required llama.cpp fork looked up the bare tag `b11443` and 404'd; the actual tag has a `-mix-<hash>` suffix. Fixed by setting `UNSLOTH_LLAMA_CPP_CONVERTER_TAG=b11541-mix-2500203`.

Files are in [`../../experiments/decision-followups/gguf-combined-step100/`](../../experiments/decision-followups/gguf-combined-step100/) (gitignored — binaries stay local, hashes below are the source of truth). See [`gguf-manifest.json`](gguf-manifest.json) for the full file list.

| File | Size | SHA-256 |
| --- | ---: | --- |
| `model-Q4_K_M.gguf` | 545,161,248 B | `abfea5f878331206f65e4117157984e14803b093b634c0ed52513c0723ef0699` |
| `mmproj-Q8_0.gguf` | 115,923,136 B | `37800be294d8c041e6146516b636414b8a75d0dd0878b08f6e0282e471790f94` |

Only Q4_K_M was exported for the main model (no Q8_0 this time); the mmproj sidecar is always kept at Q8_0. GGUF-vs-PyTorch probability parity has not been re-verified for this release the way reflex-1-beta's README documents (the 0.5015→0.4941/0.4804 flip check); repeat that spot-check before relying on the GGUF for a close decision.

## Run the GGUF Decision API

Same as reflex-1-beta — see [its README](../reflex-1-beta/README.md#run-the-gguf-decision-api) for the `llama-server` / `/v1/systemone` invocation. Point `-m` at this release's `model-Q4_K_M.gguf`.

The Qwen base model and training data sources declare Apache-2.0 (general mixture) / see dataset provenance in the handoff doc for the phone-routing data. Weights modified by decision-head and LoRA training on 2026-10-13. Not published; no public upload performed.

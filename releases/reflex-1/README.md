# reflex-1

**Renamed 2026-10-14:** this is the second ReflexModels decision-head model overall, but since the original (single-domain, typed-decisions-only) model was never published, it was renamed `reflex-1-beta` and this combined model takes the `reflex-1` name instead.

Built with Unsloth's `FastDecisionModel`/`DecisionTrainer` (the "Clef" architecture: LoRA-adapted causal backbone plus a trainable classification head for `choice`/`noul`/`score` questions). Supersedes the unpublished single-domain [reflex-1-beta](../reflex-1-beta/README.md). Research release available in a private Hugging Face repository for desktop testing.

Trained on a **combined phone-routing + general mixture** (781 phone examples + 781 class-preserving general examples: 391 typed-decisions, 195 BANKING77, 195 CLINC150; equal domain weights), independently from base `unsloth/Qwen3.5-0.8B`, LoRA rank/alpha 16, LR `2e-4`, seed 3407. Two independent combined runs were trained (80 steps and 120 steps); the **120-step run's checkpoint-100** was selected on validation and is the model exported here.

## Validation and held-out results

| Checkpoint | Phone val | General val | Equal-domain mean |
| --- | ---: | ---: | ---: |
| Combined 80-step (best) | 98.89% | 74.49% | 86.67% |
| Combined 120-step, step 80 | 98.89% | 77.35% | 88.12% |
| **Combined 120-step, step 100 (selected)** | **98.89%** | **78.39%** | **88.64%** |
| Combined 120-step, step 120 | 98.89% | 77.80% | 88.34% |

Calibration fitted and OK (`calibration_ok: true`); post-calibration ECE `.0377` vs pre-calibration `.0511`. These are validation scores used to select checkpoint 100. Held-out evaluation completed on 2026-10-14 using the restored PyTorch adapter: **60.70% typed-decisions (2,000 decisions across 400 cases), 74.89% BANKING77 (462 cases), and 81.11% CLINC150 (450 cases)**. All tests had zero skipped/truncated decisions. Typed complete-record accuracy is 10%. Official general results measure supervised in-domain adaptation. See [full protocol, comparisons and verified backup](../../experiments/decision-followups/HELDOUT_EVALUATION_20261013.md). MOBILE_ROUTING9 A32 GGUF evaluation is complete at 91.11% (164/180), with 94.44% per-case agreement to PyTorch; matched app-router evaluations are complete: uncached Qwen55.56% at41.14s; cached Qwen54.44% at35.40s. Reflex-1 mean31.36s. General scores above remain GPU results.

Known leakage/interpretation caveats (phone validation reuse across pilots, lexical overlap, in-domain-not-zero-shot general sets) are documented in the same held-out-evaluation doc — read it before quoting the phone 98.89% as a generalization claim.

The saved Qwen3.5-0.8B decision-model reference scored 8.44% on BANKING77 and 4.44% on CLINC150; Reflex-1 scored 74.89% and 81.11% accuracy on the same held-out subsets. The base reference had a freshly initialized, unseeded decision head, so it is not a conventional generative zero-shot baseline. The model card compares only base Qwen3.5-0.8B, Reflex-1 and pinned Laya English on MOBILE_ROUTING9, typed-decisions, BANKING77 and CLINC150. Larger decision models are acknowledged in related work.

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

Only Q4_K_M was exported for the main model (no Q8_0 this time); the mmproj sidecar is always kept at Q8_0. Full MOBILE_ROUTING9 comparison now finds 94.44% route agreement (10 changed predictions), despite equal aggregate accuracy of 91.11%. Mean per-case maximum absolute option-probability drift is 0.0654, and the largest is 0.8273. Quantization/runtime differences can therefore materially change confidence; do not transfer confidence thresholds without separate validation. No calibration was fitted on this test.

## Run the GGUF Decision API

Same as reflex-1-beta — see [its README](../reflex-1-beta/README.md#run-the-gguf-decision-api) for the `llama-server` / `/v1/systemone` invocation. Point `-m` at this release's `model-Q4_K_M.gguf`.

The Qwen base model and typed-decisions declare Apache-2.0; BANKING77 declares CC-BY-4.0 and CLINC150 CC-BY-3.0. The general mixture is not uniformly Apache-2.0. See [pinned source licenses](../../experiments/decision-followups/general-data/source-revisions.json) and the handoff for phone-data provenance; author contributions are Apache-2.0 with upstream attribution retained. Weights modified by decision-head and LoRA training on 2026-10-13. Privately uploaded for desktop testing; no public publication performed.

## MOBILE_ROUTING9 - 2026-10-15

On 180 fresh synthetic, human-validated cases (20 per route), the saved PyTorch adapter scores **91.11% accuracy (164/180), 90.84% macro-F1**. Zero skipped/truncated decisions. Battery recall 65%, device-info recall 75%; everyday requests 97.78%, indirect/boundary groups 84.44% each. [Per-route scores, errors, provenance and limits](../../experiments/decision-followups/fresh-phone-test-180/RESULTS.md). A32 Q4_K_M also scores 164/180 (91.11%), macro-F1 91.06%, warm p50/p95 32.34/33.87 seconds, with zero request errors. This retains aggregate accuracy while changing ten individual predictions. Matched app comparisons are complete (55.56% uncached,54.44% cached); see the A32 report for latency, resource traces and limits.

## Private Hugging Face transfer - 2026-10-15

Repository: https://huggingface.co/cbarkinozer/Reflex-1 (private). Upload revision: `157a0d6ed5a218352bcd5b6bed57dbef090b151d`. Remote sizes and LFS SHA-256 values matched the local release manifest. Adapter, joint decision head, tokenizer/processor/calibration/configs are under `adapter/`; model and projector GGUF files under `gguf/`. The standalone merged PyTorch model is **not included**: saved merged weights were absent from the verified artifacts. Adapter loading still requires the base model. The dataset was uploaded separately; see the current release-status section below.

On desktop, authenticate with your Hugging Face account and download:

```powershell
hf auth login
hf download cbarkinozer/Reflex-1 --revision 157a0d6ed5a218352bcd5b6bed57dbef090b151d --local-dir Reflex-1
```

For only A32 artifacts, add `--include "gguf/*"`. Verify hashes against `gguf-manifest.json` before benchmarking. Private token stays in your local credential store or env; never commit it.

## Mobile benchmarks and Hub model card

The source for the Hugging Face model card is [HF_MODEL_CARD.md](HF_MODEL_CARD.md). Sequential Samsung A32 evaluation is recorded in [the A32 report](../../experiments/decision-followups/a32-fresh180/RESULTS.md); only completed runs have scores. The card compares base Qwen3.5-0.8B, Reflex-1 (780M deployed text parameters) and pinned Laya English on MOBILE_ROUTING9, typed-decisions, BANKING77 and CLINC150, with hardware and training-exposure caveats. Laya scores on the same frozen cohorts are 47.22%, 36.30%, 36.36% and 55.33%, respectively; zero request errors or input truncation. Full evidence remains under `experiments/decision-followups/laya-matched-20261015/`.

## Phone dataset and release readiness

The exact [original phone splits](../../experiments/decision-followups/phone-training-781/README.md) are committed in Git: 781 train, 90 validation, 90 calibration and 88 historical test rows, with source hashes and template-group provenance. The 88-case historical test was exposed during development and is distinct from the separately frozen 180-case MOBILE_ROUTING9 test.

The dataset is now uploaded to [cbarkinozer/MOBILE_ROUTING9](https://huggingface.co/datasets/cbarkinozer/MOBILE_ROUTING9): 781 train / 90 validation / 90 calibration / 180 test, with pairwise split-isolation checks and verified export hashes. Historical test rows are excluded from that Hub release. The model's additional 781 general training examples are not included in this phone dataset. [Dataset card source](dataset/HF_DATASET_CARD.md).

Both author contributions are licensed under Apache-2.0, with upstream attribution preserved in LICENSE/NOTICE files. The [final release check](FINAL_RELEASE_CHECK.md) and [license upload receipt](hf-license-update-20261015.json) record the completed audit. Both Hub repositories were private at that audit; the author controls subsequent public visibility. No Git push was performed by the agent.

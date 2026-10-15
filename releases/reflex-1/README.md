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

Calibration fitted and OK (`calibration_ok: true`); post-calibration ECE `.0377` vs pre-calibration `.0511`. These are validation scores used to select checkpoint 100. Held-out evaluation completed on 2026-10-14 using the restored PyTorch adapter: **96.59% historical phone (85/88), 60.70% typed-decisions (2,000 decisions across 400 cases), 74.89% BANKING77 (462 cases), and 81.11% CLINC150 (450 cases)**. All tests had zero skipped/truncated decisions. Typed complete-record accuracy is 10%. Historical phone has prior development exposure; official general results measure supervised in-domain adaptation. See [full protocol, comparisons and verified backup](../../experiments/decision-followups/HELDOUT_EVALUATION_20261013.md). GGUF parity and on-device benchmarking remain pending.

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

The Qwen base model and typed-decisions declare Apache-2.0; BANKING77 declares CC-BY-4.0 and CLINC150 CC-BY-3.0. The general mixture is not uniformly Apache-2.0. See [pinned source licenses](../../experiments/decision-followups/general-data/source-revisions.json) and the handoff for phone-data provenance; finalize redistribution terms and attribution before publication. Weights modified by decision-head and LoRA training on 2026-10-13. Privately uploaded for desktop testing; no public publication performed.

## Fresh phone routing test - 2026-10-15

On 180 fresh synthetic, human-validated cases (20 per route), the saved PyTorch adapter scores **91.11% accuracy (164/180), 90.84% macro-F1**. Zero skipped/truncated decisions. Battery recall 65%, device-info recall 75%; everyday requests 97.78%, indirect/boundary groups 84.44% each. The historical phone score of 96.59% does not describe this harder cohort. [Per-route scores, errors, provenance and limits](../../experiments/decision-followups/fresh-phone-test-180/RESULTS.md). GGUF and on-device results remain pending.

## Private Hugging Face transfer - 2026-10-15

Repository: https://huggingface.co/cbarkinozer/Reflex-1 (private). Upload revision: `157a0d6ed5a218352bcd5b6bed57dbef090b151d`. Remote sizes and LFS SHA-256 values matched the local release manifest. Adapter, joint decision head, tokenizer/processor/calibration/configs are under `adapter/`; model and projector GGUF files under `gguf/`. The standalone merged PyTorch model is **not included**: saved merged weights were absent from the verified artifacts. Adapter loading still requires the base model. Dataset upload remains separate.

On desktop, authenticate with your Hugging Face account and download:

```powershell
hf auth login
hf download cbarkinozer/Reflex-1 --revision 157a0d6ed5a218352bcd5b6bed57dbef090b151d --local-dir Reflex-1
```

For only A32 artifacts, add `--include "gguf/*"`. Verify hashes against `gguf-manifest.json` before benchmarking. Private token stays in your local credential store or env; never commit it.

## Phone training data on GitHub

The exact [781-row phone training split](../../experiments/decision-followups/phone-training-781/README.md) is committed in this repository. The separate Hugging Face dataset repository was deleted at the user's request. Model weights remain private on Hugging Face. Validation/calibration/test rows are excluded from this training dataset; the combined model additionally used 781 general examples.

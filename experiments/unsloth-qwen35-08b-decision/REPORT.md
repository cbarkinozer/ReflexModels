# Qwen3.5-0.8B decision model: completed reproduction

The adapted Unsloth notebook trained successfully on the Vast.ai RTX 4090. Independent typed-decisions test accuracy improved by 31.3 percentage points. The saved adapter was reloaded and evaluated in a fresh process without refitting calibration.

| Measurement | Accuracy | ECE |
|---|---:|---:|
| Before training, fresh decision head | 34.40% | 0.01848 |
| After training and holdout calibration | 65.70% | 0.02777 |
| Saved adapter, independent reload | 65.75% | 0.02729 |

Test size: 400 source records / 2,000 decisions. Training used 1,120 records, with 80 held out for calibration. Normalized source-state train/test overlap was zero; no decisions were skipped or truncated. The reload difference is one decision out of 2,000. The initial baseline uses a newly initialized decision head and is not the phone's prompted Qwen baseline.

## Recipe and performance

Qwen3.5-0.8B, typed-decisions/all, 4-bit LoRA rank/alpha 16, backbone LR 2e-4, head LR 1e-4, batch 8 with accumulation 4, context 2048, seed 3407, 60 steps (1.714 epochs), warmup 10, cosine schedule, weight decay 0.01, bf16. Training took 150.98 seconds (2.52 minutes). Peak Torch reserved memory was 2.938 GiB; runtime GPU usage includes additional allocations. The adapter/head is approximately 113 MB and the merged model approximately 1.7 GB.

Raw holdout accuracy was 66.75%; calibration retained accuracy and reduced holdout ECE from 0.13244 to 0.06161. Reported library loss uses soft targets; it is not labeled as hard-label NLL. No Brier score was measured.

Model revision: `23c69c53358a07516b5827588b3fdb12ae78fd65`.
Dataset revision: `6a7927fa3c8ac114e29f5300eacb86f6f37a6447`.
Unsloth commit: `a70b2a8f7b88c3b8492409efef1dce3d670787f6`; zoo commit: `ea7c1b2d9eb5daf0bc6036bf68032795868f2d74`. Exact packages, training arguments and source hashes are retained in the backup.

## Validation and limits

The two-step smoke completed with finite loss and adapter reload. Separate 32-record tiny-overfit diagnostics (60 and 300 steps, different learning rates) reached 46.875% from 40% and did not memorize the data. This unresolved diagnostic is retained; the unchanged notebook recipe subsequently showed substantial learning on the independent test. Optional Mamba/FLA kernel installation did not succeed; the actual run used the validated Torch/Unsloth fallback.

This is a typed-decisions notebook adaptation, not the blog's mixed 12-source recipe. It does not reproduce the advertised 73% typed-decisions score. The main run did not measure BANKING77 or CLINC150; added evaluations are reported below. Turkish quality, phone latency, memory, thermals and tool-routing performance remain unmeasured. Only one seed was run. These results support proceeding to runtime experiments, but do not establish an Android improvement.

## Artifacts and tracking

The local verified backup is `runs/pod-backup/extracted/`: all 77 inventory file hashes passed, including the saved adapter/head, raw metrics, exact run sources and offline W&B records. A W&B diagnostic-log symlink was materialized from the pod before verification. The compact transfer archive SHA256 is `6c2539de33cbfc0dacfe14d8b114381088bee4469496e2393684b72a9c431a4c`. Model weights and raw runs are Git-ignored. The merged model is now backed up locally at `runs/pod-backup/merged`; the final optimizer checkpoint is at `runs/pod-backup/checkpoints`. Both were checked against pod-generated SHA256 inventories.

W&B project: `reflexmodels`; experiment: `qwen35-08b-decision-typed-decisions-lora16-lr2e4-60steps`; offline run ID: `4khq3xdg`. The supplied key returned HTTP 401, so actual fine-tuning was logged offline. Smoke and diagnostic runs were not logged to W&B. The replacement key authenticated successfully; the original run was synced to https://wandb.ai/c-barkinozer/reflexmodels/runs/4khq3xdg without creating another training run.

Remaining work: a separate validated GGUF/head export and inference-parity check; bounded-decision integration into pocket-agents-lab and matched phone evaluation; then an independent Mac mini experiment. The current training stage is complete. The pod has not been stopped.

## Additional balanced benchmark evaluation

| Dataset | Official test sample | Fresh-head baseline | Trained adapter | Trained ECE |
|---|---:|---:|---:|---:|
| typed-decisions | 400 records / 2,000 decisions | 34.30% | 65.60% | 0.02655 |
| BANKING77 | 462 requests: 6 per each of 77 intents | 8.23% | 44.59% | 0.28004 |
| CLINC150 | 450 requests: 3 per each of 150 intents | 4.67% | 47.11% | 0.33968 |

Scoring used published gold labels with no human evaluation. Typed-decisions contains four workflows with 100 test records each. Sampling seed was 3407. All 77 or 150 intent candidates were supplied, described using the original label names with underscores replaced by spaces. CLINC out-of-scope requests were excluded. The original typed-only model and its saved calibration were reused without tuning or refitting on these tests.

The initial 2,048-token evaluation could not fit CLINC's 3,042-token option schema and skipped all 450 CLINC rows. That attempt and its exact source/logs were retained, and final evaluation used 4,096 tokens for both model stages and all datasets, with zero skipped or truncated decisions. Training context remains 2,048. The fresh-head baseline is a separate initialization; the final typed repeat differs from the independent 2,048-token adapter reload by three of 2,000 decisions. Neither run substitutes for the other.

These tests establish transfer gains, not the blog's exact scores. Our main holdout accuracy is 66.75%, versus its reported 78%; its typed/BANKING/CLINC scores are 73%/74%/76%. The blog describes mixed-source training with rank 64/one epoch, but recommends rank 16/two epochs elsewhere. We followed the rank-16, 60-step typed-only notebook after changing its 4B model to 0.8B. Data mixture, training duration, rank, question wording, candidate descriptions and held-out samples all differ or are unspecified. Hyperparameter mismatch is plausible but not isolated. Ordinary text-generation temperature, top-p and token-generation limits do not govern the decision head.

A stricter reproduction requires the benchmark training mixture and sampling weights, exact split/sample IDs, question construction, and actual run configuration. Subsequent training comparisons must use separate validation data and keep these tests frozen. The high banking/CLINC ECE shows that typed-decisions calibration should not be assumed to transfer across tasks.

Benchmark sources: [Unsloth training article](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth), [official BANKING77 source](https://github.com/PolyAI-LDN/task-specific-datasets), [CLINC dataset](https://huggingface.co/datasets/clinc/clinc_oos). Exact source revisions, selected rows, labels and full decision schemas are preserved in the local backup.

## Structured inference demonstration

The trained adapter answered the supplied billing request through `/v1/systemone` using a custom localhost HTTP adapter over `FastDecisionModel.predict`. Three requests validated choice/noul/score schemas and normalized probabilities. `model.generate` was replaced with a function that raises, and requests still succeeded; no text-generation path was used. The third request took 0.110 seconds on the RTX 4090. This is not a latency benchmark or a CPU/phone measurement.

The result chose billing with probability 0.9491, urgency level 2 with probability 0.8264, and refund=True with probability only 0.5039. The weak refund margin is retained rather than presented as confident success. Request/response JSON and service sources/logs are backed up. This proves our structured inference path operates, not that the official Studio server, Mac runtime, Android runtime or decision thresholds have been validated.

W&B authentication was retested with the exact user-supplied key; it matched the original key and again received HTTP 401 from api.wandb.ai/graphql under SDK 0.25.0. Only the non-secret authentication status is recorded in `wandb-auth-status.json`.

The merged model also reloaded successfully in 4-bit mode and answered the same three questions without generation. All top answers matched the adapter example, but probabilities differed by up to 0.0493 after merging and re-quantization. This is a single-request load check, not full dataset parity; converted runtime quality still needs its own evaluation before Android use.


## reflex-1 release preparation

The trained model was named reflex-1 at the time (renamed reflex-1-beta on 2026-10-14 once a later combined model took the reflex-1 name). See ../../releases/reflex-1-beta/README.md. Replacement W&B authentication and original-run cloud sync succeeded. GGUF export uses the dedicated decision exporter; its initial automatic llama.cpp release download returned 404, so a direct upstream source checkout is used instead.

GGUF Q8_0 and Q4_K_M export completed using llama.cpp revision aa94f2086147cd4bfbcff818ce998dffce5fcb39. Both passed one CPU structured request (choice/noul/score), output_tokens=0. A close refund decision flipped: merged PyTorch yes=0.5015, Q8=0.4941, Q4=0.4804. Billing/urgency agreed. This is an API/load smoke, not full accuracy parity. CPU request times were 1.54s / 1.32s on the pod; Android performance remains unmeasured. Sources, build/export/failure logs, GGUF inventory, calibration and raw response validation are preserved.

Cost snapshot and calculation are preserved in [COST.md](COST.md) and cost.json. At 2h 7m × $0.363/hour, estimated base cost is $0.76835, excluding internet and any later runtime. Final billing/shutdown remain unconfirmed. Future closeout requirements are in ../../EXPERIMENT_CLOSEOUT.md. reflex-1 remains unpublished at the user’s request.

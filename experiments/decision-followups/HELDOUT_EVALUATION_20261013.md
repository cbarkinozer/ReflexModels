# Held-out decision evaluation — 2026-10-13

This document records the final test protocol and results for the independent Qwen3.5-0.8B fine-tunes. It is updated as the GPU run finishes. The operational handoff, exact commands, logs and cost are in [VAST_FINE_TUNE_HANDOFF_20261013.md](VAST_FINE_TUNE_HANDOFF_20261013.md).

## Candidate selection before test access

- Phone-only: 60 steps, validation `89/90 = 98.89%`.
- General-only: 80 steps, native validation decision accuracy `76.39%` after best-checkpoint reload.
- Combined 80 steps: phone validation `98.89%`, general validation `74.49%`, equal-domain mean `86.69%` after best-checkpoint reload.
- Combined 120-step follow-up: **complete**. Step 80: phone `98.89%`, general `77.35%`, equal-domain mean `88.12%`. Step 100: phone `98.89%`, general `78.39%`, equal-domain mean `88.64%` — **best checkpoint, selected**. Step 120: phone `98.89%`, general `77.80%`, equal-domain mean `88.34%` (slightly worse than step 100, i.e. extra training did not help further). **Decision: use checkpoint-100 of the 120-step run** (`runs/combined-balanced-scratch-120step-s3407/adapter`, already saved as the best-checkpoint reload) as the selected combined candidate — it beats both the 80-step run's `86.67%` and step 120. The final test has not been used for this comparison.
- Base reference: untouched `unsloth/Qwen3.5-0.8B`, evaluated with the same schema and test cases.

## Frozen test protocol

The runner `pocket-agents-lab/runs/decision-followups-vast-20261013/run_heldout_evals.sh` will first freeze and hash these test files after model selection, then evaluate base, phone-only, general-only and the selected combined adapter on identical files with 4096-token context and batch size 8:

| Test | Intended cases | Meaning |
| --- | ---: | --- |
| Historical phone | 88 usable | Nine-route phone routing; previously seen during project development |
| typed-decisions official test | 400 | Multi-question supervised benchmark test |
| BANKING77 official test subset | 462 | Six per class, 77 classes |
| CLINC150 official test subset | 450 | Three per in-domain class; OOS excluded |

The official general source revisions are pinned in [GENERAL_DATA.md](GENERAL_DATA.md). Test subset seed is 3407. The preparation script asserts zero normalized exact state overlap with phone and general training and records source hashes and selected IDs in its manifest. It cannot guarantee no semantic paraphrase overlap or exclude exposure during foundation-model pretraining. BANKING77 and CLINC150 are **in-domain supervised adaptation** tasks here, not zero-shot tests.

## Leakage and interpretation limits

The phone export has no normalized exact train/validation duplicate and no overlap in automatic same-route template groups. A fresh lexical audit found two high-similarity train/validation pairs: “Is the battery temperature safe?” / “Is the battery temperature normal?” and “What are my storage stats?” / “What is my storage status?”. Eleven of 90 validation queries have a nearest training query lexical score ≥0.8 under the audit's max(Jaccard, SequenceMatcher) rule. This is not proof of deliberate data leakage, but it limits independence.

The same 90 phone validation IDs were used in an earlier 300-review pilot and later dataset comparisons. As a result, 98.89% is a development-set score that may be optimistic from repeated exposure and selection. The historical phone test was also exposed earlier. A newly authored, sealed phone cohort is needed before making a strong generalization claim. A character TF-IDF baseline scored 85/90 (94.4%) on the same phone validation set, indicating that many examples are separable by wording even without the fine-tuned model.

Calibration used its own disjoint split; the fitted calibration score is not a new external test. Test performance should be reported per source and per decision, not as one pooled number dominated by class counts. The evaluator currently writes aggregate native accuracy, ECE, loss, record accuracy, row counts and build reports; it does not save per-case predictions.

## Results ? completed 2026-10-14

All four models were evaluated on identical frozen files after validation-only checkpoint selection. These are native decision accuracies; typed-decisions has 400 cases containing 2,000 questions, while the other tests have one decision per case.

| Model | Historical phone | typed-decisions | BANKING77 | CLINC150 |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3.5-0.8B | 13.64% | 34.10% | 8.44% | 4.44% |
| Phone-only | 97.73% | 48.35% | 53.46% | 52.00% |
| General-only | 77.27% | 58.55% | 71.43% | 78.89% |
| Selected combined (reflex-1) | 96.59% | 60.70% | 74.89% | 81.11% |

All 16 model/source evaluations report zero skipped and zero truncated decisions. Counts: phone 88, typed 400 cases / 2,000 decisions, BANKING77 462, CLINC150 450. Combined typed-decisions record accuracy (all five questions correct) is 10%, so its 60.70% per-decision score does not mean 60.70% of complete records were correct.

The selected combined model leads these tested models on the three general sources and gets 85/88 phone cases correct; phone-only gets 86/88. A one-case phone difference does not establish superiority. Phone validation's 98.89% should still not be treated as independent evidence: the historical test also has prior exposure. General official tests show useful adaptation, with substantial remaining errors, especially on typed decisions. No checkpoint or hyperparameter was changed based on these tests. Further model improvements need a new sealed test cohort.

The base reference uses FastDecisionModel's decision API and head initialization, rather than a prompted generative zero-shot classifier. The evaluation runner does not explicitly seed initialization; preserve the measured reference but do not present it as a reproducible generative zero-shot baseline. No per-case predictions were saved, so the aggregate results alone cannot identify individual mistakes or per-class bias.

Combined ECE: phone 0.03610, typed 0.04233, BANKING77 0.07579, CLINC150 0.05715. Raw losses, record accuracies and build reports are retained in the backup. These results apply to the restored PyTorch adapters; GGUF parity and on-device timing remain pending.

Frozen file SHA-256:

- typed: `1d6c24f9709b62451ece105df85b8f293a4cde9a5a8b73721928b62876ad8365`
- BANKING77: `3c7873d2de8b9bad9445c3acf72e85802a499ed059c44626121cc4fdd3b510c0`
- CLINC150: `0a484b00b94b66d5a7aeefb564eaab9393cc737a4b5078690055e2778e8aa3bd`

Verified local backup: `pocket-agents-lab/runs/decision-followups-vast-20261013/heldout-20261014/heldout-results-20261014.tar.gz`, 313313 bytes, SHA-256 `5ecd36269dda3392f76f05b8103a9e0d57e57552f90cab66761bf1d2c1b02e2f`. Includes raw JSON, full evaluation/setup/download logs, frozen test files/manifests, executed tools, launcher and resolved environment. Extracted results are alongside it.

## Evaluation restart ? 2026-10-14

A fresh RTX 4090 24 GB instance is reachable at `ssh -p 12238 root@ssh4.vast.ai`. Restoring selected phone/general/combined adapters from the verified local project backup; no retraining or checkpoint selection is performed. The combined candidate remains step 100. Evaluation-only work creates no W&B runs. The base model, phone adapter, general adapter and selected combined adapter will use the same frozen test files.

Transfer bundle: `pocket-agents-lab/runs/decision-followups-vast-20261013/heldout-inputs-20261014.tar.gz`, 260620163 bytes, SHA-256 `c6c28b6263b18b2b635c4780286d4dbb662bef055fe880bb87e4de3233af1940`. This contains data, scripts and selected adapters/heads; full training backups remain local. Setup log: `/workspace/setup-evaluation.log`; orchestration log: `/workspace/heldout-launch.log`. The launcher verifies the bundle hash before extracting and running tests. Original Unsloth/Unsloth Zoo git revisions and Torch 2.8.0 are restored. Exact resolved environment is saved separately for this evaluation.

Status: all evaluations complete and verified local backup downloaded. Instance `55976079` remains running and idle. Snapshot at approximately 19:15 UTC: duration 1144.53 seconds, rate $0.53037037/hour, elapsed-rate estimate $0.16861. This is not a final invoice; network charge totals are unconfirmed and billing continues. Sanitized provider snapshot is saved locally as `heldout-20261014/evaluation-cost-snapshot.json`. Keep this instance running unless the user explicitly authorizes shutdown.

## Verified pod closeout backup ? 2026-10-14

The complete evaluation workspace backup is downloaded at `pocket-agents-lab/runs/decision-followups-vast-20261013/heldout-20261014/heldout-closeout-20261014.tar.gz`: 260852840 bytes, SHA-256 `883959885b9280506688ab33fdeac3560fdef13ddcf4ed37d9105ff750b98a90`. The local archive matches the remote hash; every one of the 94 inventoried files also passed byte-count and SHA-256 verification. Includes all selected adapters, decision heads, tokenizers, calibration/config files, prepared training/validation/calibration/test data, manifests, executed source, logs, environment freezes, GPU details and sanitized provider snapshots. No new training occurred, so optimizer checkpoints and original W&B records remain in the earlier full training archive. Excludes credentials, virtualenv and reproducible download/compiled caches. The earlier base-model cache and GGUF files remain local. No unique evaluation output depends on this pod anymore.

Closeout provider snapshot at 19:31:20 UTC: instance 55976079, RTX 4090, host 109053, machine 36225, duration 2096.77 seconds, total rate $0.53037037/hour (base $0.49333333 plus storage $0.03703704), elapsed-rate estimate about $0.30890. This is not the final invoice; network billed counters are retained without assuming their units, and charges continue until the user closes the pod. The instance was left running; no stop/destroy action was taken.

Additional exact-overlap audit on the downloaded backup found **zero normalized text overlap** between each official test (typed, BANKING77, CLINC150) and every restored phone/general training, validation and calibration split. Report: `heldout-20261014/exact-overlap-closeout-audit.json`. This does not check semantic/template overlap and cannot rule out foundation-model pretraining contamination. Public benchmark scores do not establish zero-shot generalization; the general adapter trained on these datasets' training splits and familiar intent labels.

## Fresh phone test frozen ? 2026-10-15

All 180 fresh reviews are submitted and accepted, exactly 20 per route. The separate [frozen test record](fresh-phone-test-180/README.md) documents schema, hashes, overlap audit and review limitations. No model predictions have been obtained. This cohort remains excluded from training, calibration and checkpoint selection. Raw prior benchmark JSON, environment and frozen official test files are committed in `heldout-results-20261014/`; full logs/models remain in the verified local archive.

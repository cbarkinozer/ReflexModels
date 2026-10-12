# General decision data preparation

Prepared locally on 2026-10-11. No GPU rental, fine-tuning or W&B run started.

The frozen training data is at `data/prepared/general-routing-v1-balanced/`.
Corpus files are ignored; committed manifests, selected IDs, hashes, preparation
logs and environment records are in `general-data/`. An earlier local
`general-routing-v1/` preparation is superseded by the balanced version.

| Split | Cases | Decisions | typed-decisions | BANKING77 | CLINC150 |
|---|---:|---:|---:|---:|---:|
| Training | 2,240 | 6,720 | 1,120 | 560 | 560 |
| Validation | 2,529 | 2,689 | 40 | 993 | 1,496 |
| Calibration | 1,287 | 1,447 | 40 | 497 | 750 |

Training requested source ratios are 1:0.5:0.5, achieved as 50%/25%/25%.
Sampling uses no replacement, retains every intent class and balances within
each source. These counts describe the prepared pool; actual optimizer exposure
must still be recorded during training. CLINC OOS is excluded.

## Source corrections

The previously listed three short revision IDs returned HTTP 404. Replacement
commits are verified and use the original experiment's sources:

- typed-decisions: `6a7927fa3c8ac114e29f5300eacb86f6f37a6447`, config `all`, Apache-2.0.
- BANKING77 upstream: `57ec275d8078af65b7731c2a98be812d844a6d6b`, CC-BY-4.0.
- CLINC: `155b9c710419136e17307b80d0a13e68cd46b4ec`, config `plus`, CC-BY-3.0.

BANKING77 is downloaded directly from pinned upstream CSVs. Its HF dataset
loader references a mutable upstream branch, which a loader revision would not
freeze. Download URLs and payload hashes are recorded.

Conversion now preserves all typed questions and soft gold distributions. Intent
records include gold labels/probabilities. IDs use source split/index or original
typed IDs, avoiding process-dependent Python hashes.

## Isolation and historical exposure

Official test states and all existing phone states are exclusion inputs only.
Normalized duplicate/protected-overlap exclusion removed 115 records. Identity
checks passed across prepared splits and against original reflex-1 training.
This is not a semantic/paraphrase decontamination claim.

The pinned Unsloth `split_holdout` algorithm, with seed 3407 and a 400-decision
cap, originally held out 80 five-question typed cases. Original source row
order/content is checked against the saved training snapshot. Those 80 cases
are split 40/40 for validation/calibration; the 1,120 original training cases
never enter either partition. These 80 cases were previously evaluated and
calibrated, so their scores are not pristine unseen evidence.

Original BANKING77 and CLINC150 historical evaluation selections were copied
byte-for-byte; their hashes are in the manifest. No new general test cohort was
sampled. Phone files were preserved. A new reviewed, sealed phone cohort remains
necessary for final claims after iterative development.

## Reproduction

Use the recorded CPU environment, or install `datasets` in an isolated environment.
Run from pocket-agents-lab with a new output directory (existing ones are refused):

```sh
python tools/prepare_general_mixture.py \
  --output-dir /path/to/ReflexModels/data/prepared/general-routing-v1-balanced \
  --hf-cache /path/to/hf-cache \
  --historical-dir /path/to/ReflexModels/experiments/unsloth-qwen35-08b-decision/runs/pod-backup/extracted
```

The recorded seed is fixed at 3407 to reconstruct original checkpoint exposure.
Preserve this data directory and its manifests when copying inputs to a pod.
Copy `decision_training_utils.py` alongside all three training scripts.

## Training fixes and checks

The shared training helper explicitly enables the loaded Clef `model.head` and
existing LoRA parameters. Checkpoint selection uses native
`FastDecisionModel.evaluate` accuracy every 20 steps, preserving multi-question
records and explicit gold labels. Generic Trainer array metrics can count cases
instead of questions and mishandle soft probability targets. Mixed-schema
option indices must not be treated as semantic classes for macro-F1.

Smoke inference is limited to 16 validation/calibration cases per domain, with
32 training cases/two steps and no W&B. Validation-only native scoring selects
checkpoints; test scoring remains a separate final operation.

Eight CPU regression tests passed, covering gold conversion/stable IDs, soft
targets/padding, explicit gold labels, mixed-schema metrics, multi-question native
evaluation, deterministic disjoint splits, class-preserving mixture sampling,
and protected-state removal. Dataset construction, all choice mappings, full
77/150 class coverage, hashes, syntax compilation and CLI help also passed.

GPU dataset building/tokenization, adapter/head loading, actual optimization,
calibration and saved/reloaded inference have not been validated. These remain
smoke requirements. Both previous pod SSH endpoints failed (direct timeout,
proxy connection refused); no current rental rates or cost were assumed.

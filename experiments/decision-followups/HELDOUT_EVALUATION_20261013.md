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

## Results

Pending completion of the validation-selected candidate and the test runner. Fill this table from the four `runs/eval-*-s3407/summary.json` files; retain the raw JSON and logs in the downloaded archive.

| Model | Historical phone | typed-decisions | BANKING77 | CLINC150 |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3.5-0.8B | pending | pending | pending | pending |
| Phone-only | pending | pending | pending | pending |
| General-only | pending | pending | pending | pending |
| Selected combined | pending | pending | pending | pending |

After evaluation, record exact case counts, decision counts, skipped/truncated records, source file hashes, selection reason, observed errors/limitations, artifact archive path and SHA-256 here. The pod remains running per the user's latest instruction; do not destroy it.

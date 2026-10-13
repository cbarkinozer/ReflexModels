# Three decision-model follow-up runs

> **Current status (2026-10-13):** See [VAST_FINE_TUNE_HANDOFF_20261013.md](VAST_FINE_TUNE_HANDOFF_20261013.md). The user chose three independent runs from base Qwen3.5-0.8B; the checkpoint-based starting-point plan below is historical. Three runs have completed, and one validation-only combined follow-up is running.

Requested 2026-10-11. Status: planned; none of these training runs has started.
Reuse one pinned GPU environment, downloaded datasets and evaluation artifacts.

| Run | Starting weights | Training data | Main question |
| --- | --- | --- | --- |
| `qwen35-08b-reflex1-phone-routing-v1-lora16-s3407` | Saved reflex-1 checkpoint | Phone-routing training pool only | Does domain adaptation beat Qwen on the same frozen 90 phone cases? |
| `qwen35-08b-reflex1-general-mix-v1-lora16-s3407` | Same saved reflex-1 checkpoint, independently loaded | Official training partitions of typed-decisions, BANKING77 and CLINC150, converted to decision schemas | Can a broader mixture improve general decisions and preserve useful transfer? |
| `qwen35-08b-reflex1-combined-v1-lora16-s3407` | Same saved reflex-1 checkpoint, independently loaded | General training mixture plus reviewed phone training data | Can one model retain general capability and specialize in phone routing? |

The second run does not continue from the phone-trained model. Keeping the same
starting checkpoint makes the three outputs interpretable as separate adaptations.
Retain reflex-1 as the before-adaptation reference on every evaluation.
Record combined-run sampling weights and per-source example exposure: compare
phone exposure as well as total optimizer steps. Do not let a large general pool
silently overwhelm phone examples. Choose mixture weights on validation only.

## Laya comparison and iterative search

- Benchmark explicit single checkpoints: English Laya (421M), multilingual
  Laya (322M), and the typed-decisions specialist where relevant. Do not use
  automatic checkpoint routing as an unnamed single-model baseline.
- Official reference: https://huggingface.co/convaiinnovations/laya
- Community conversion: https://huggingface.co/Weidows/laya-multilingual-GGUF
  Its Q4_K_M is encoder-only; the original decision head ships separately.
  HF token IDs are required according to the conversion card. Verify complete
  inference and prediction/probability parity before A32 timing; ordinary text
  generation commands are not evidence that decisions work.
- Use identical frozen cases and semantic question/option schemas. Record
  context and option budgets and truncation, especially 77/150-option questions.
- Measure accuracy, macro-F1 and calibration with native benchmark scorers;
  compare warm p50/p95 latency, cold load time, full-process PSS and thermal
  conditions on the same device. Include head/tokenization costs in timing.
- Review synthetic gold labels before training. Split scenario/paraphrase
  families before generating variants. For a new 80/20 dataset, keep the 20%
  final test sealed and derive validation/calibration from the 80% pool.
- Existing frozen phone test remains unchanged. Its previous reported results
  make it a historical comparison set, not a wholly unseen final test for an
  extended adaptive search. Add a separately sealed cohort for final claims.
- Automated labeled evaluation enables immediate subsequent trials, using
  validation only for dataset/hyperparameter selection. Record every trial;
  open sealed tests after selecting final candidates, not after every trial.
- User wording about "10" phone datasets/routes is awaiting clarification;
  the existing schema has nine routes. Do not invent a tenth route.

## Protect evaluation data

- Preserve the phone development cases, frozen 90-case phone test, and saved
  typed-decisions/BANKING77/CLINC150 test rows and source revisions.
- Use official training partitions for general adaptation; never train on the
  benchmark test questions, answers, paraphrases generated from test questions,
  or model predictions on those questions.
- Check normalized exact overlap against every protected evaluation set. Check
  near duplicates and shared paraphrase/template families across splits; an
  exact-string check alone does not establish independence.
- Before tuning, derive validation examples from training data only. Keep
  calibration separate from checkpoint/hyperparameter selection and keep tests
  out of smoke runs and iterative training decisions.
- For the phone run, one suitable derived split is 810 training / 90 validation
  from the existing 900-record training pool, preserving class balance. The
  existing 90 calibration / 90 test files remain untouched. Save derived IDs,
  seed and hashes. Audit template families before freezing the derived split.
- Pin general dataset revisions, source IDs, option schemas, mixture weights,
  selection seed and all derived split manifests before training.

Training on the training partition of a benchmark is legitimate supervised
adaptation. Describe the resulting general evaluation as in-domain supervised
performance, not unseen-domain or zero-shot performance.

## Before spending pod time

The current phone script starts from fresh Qwen, adds new PEFT adapters, uses
calibration rows during training evaluation, evaluates test rows even in smoke
mode, omits probability calibration and disables W&B. Adapt the loading and
split handling explicitly before using it for these continuation runs; do not
silently pass an existing PEFT checkpoint through fresh-adapter initialization.

Reuse the saved reflex-1 environment pins and resolved training settings. Start
with rank 16 and seed 3407; choose training duration using validation performance
and example exposure. Sixty steps is a phone smoke-scale/full-trial starting
budget, not automatically sufficient for the larger general mixture. Record
epochs, optimizer steps, effective batch size and per-source exposure. Save the
initial loaded-checkpoint fingerprint and evaluate validation before training.

## Efficient execution and evaluation

1. Prepare all three immutable training mixtures, scripts and configs locally.
2. Set up one pod, validate checkpoint loading and run short training smokes
   using training/validation data only; no W&B and no test evaluation.
3. Run all three actual fine-tunes sequentially, each with its own named W&B run.
4. Choose checkpoints using validation only, calibrate on separate calibration
   rows, then evaluate all three outputs and reflex-1 on identical frozen phone,
   typed-decisions, BANKING77 and CLINC150 test selections. Reuse loaded data.
5. Save per-example predictions, calibration metrics, build/skip/truncation
   reports and all configs/revisions. Do not accept silently dropped examples.
6. Export Q4_K_M, verify reload/export parity and back up all three full models,
   adapters/heads, tokenizer/calibration, checkpoints, logs and manifests locally.
7. Measure relevant candidates on the A32 on the same cases/settings as Qwen;
   measure accuracy, latency, PSS, heat and cooldown. GPU inference timing is not
   a substitute for phone timing. Follow EXPERIMENT_CLOSEOUT.md for costs/backups.

The matched phone reference is Qwen raw HTTP at 55.6% / 36.8 seconds on the
90-case phone test. A small point-estimate improvement alone is not evidence of
practical reliability: inspect per-route errors and paired outcomes. Further
iterations should use validation, preserving the frozen test from selection.

One-loaded-backbone decision/text switching remains separate runtime work.
Neither run automatically enables final text generation through the Clef GGUF.
All outputs remain local and unpublished.

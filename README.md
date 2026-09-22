# ReflexModels

ReflexModels is an open research project for small semantic decision models:

```text
state + natural-language question + runtime-defined options
                           -> calibrated typed probabilities
```

Turkish is the primary specialization from the first experiment; English is a
control and transfer language. The models target CPU-friendly inference and
make decisions directly rather than generating and parsing text.

## What it is—and is not

The core API supports choices defined at inference time. Adding a new tool,
renaming a label, or supplying a new taxonomy should not require retraining.

```python
choice = Choice(
    state="Kullanıcı config.yaml dosyasını incelemek istiyor.",
    question="Hangi araç kullanılmalı?",
    options={
        "filesystem": "Yerel dosyaları oku",
        "web": "İnternette ara",
        "shell": "Komut çalıştır",
    },
)
```

This is not a claim that ReflexModels will outperform a fixed-label encoder
such as BERTurk on every static classification task. The central test is
unseen-option generalization, calibrated probabilities, shared-state
multi-question execution, and CPU viability.

## Research contract

- Turkish and English results are always reported separately.
- Turkish examples record `native`, `translated`, or `synthetic` provenance.
- Frozen benchmarks are never used for training, prompting, or tuning.
- Every run records configuration, seed, dataset versions, hardware, and
  machine-readable metrics.
- No architecture change is made before bilingual baselines exist.

See [PLAN.md](PLAN.md), [DATASETS.md](DATASETS.md),
[DECISIONS.md](DECISIONS.md), and [RESULTS.md](RESULTS.md).

## Layout

```text
benchmarks/       Frozen evaluation data and annotation protocol
configs/          Versioned experiment configurations
results/          Reproducible machine-readable run outputs
scripts/          Benchmark and data-validation entry points
src/              Shared evaluation primitives
```

## Status

Milestone 0: bilingual baseline and measurement foundation.

## CPU fixed-label baseline

`run_fixed_label_baseline.py` is the reproducible BERTurk/XLM-R reference
runner. It only fine-tunes and evaluates on CPU, writes raw and
temperature-calibrated test metrics separately, and reports single-request CPU
latency. Run both declared overlap policies; the sensitivity policy removes
train/dev utterances that text-match test but never changes the test split.

```powershell
.\.venv\Scripts\python.exe scripts\run_fixed_label_baseline.py `
  --config configs\baselines\massive_fixed_label.example.json `
  --run berturk_tr `
  --model-path data\raw\models\dbmdz-bert-base-turkish-cased `
  --overlap-policy report_only `
  --output-dir results\berturk_tr_report_only
```

The runner requires the compatible CPU environment recorded in
[`requirements-cpu.txt`](requirements-cpu.txt). Checkpoint files remain local
and are deliberately excluded from Git.

Before a full CPU training run, exercise the entire path with a short smoke
run. It trains, saves and reloads an epoch checkpoint, calibrates on validation,
and writes test predictions and metrics:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_fixed_label_baseline.py `
  --config configs\baselines\massive_fixed_label.example.json `
  --run berturk_tr `
  --model-path data\raw\models\dbmdz-bert-base-turkish-cased `
  --overlap-policy report_only `
  --output-dir results\berturk_tr_smoke `
  --epochs 1 --batch-size 4 --max-length 64 --num-threads 4 `
  --limit-train 8 --limit-dev 8 --limit-test 8
```

Every full epoch writes a checkpoint under `results/<run>/checkpoints/` before
the next epoch starts. If evaluation fails, rerun only evaluation with
`--evaluate-checkpoint <checkpoint-file>` and the original run arguments.

## V0 causal-LM option scoring

`run_v0_causal_choice.py` runs the non-generative duplicated-branch oracle
against a local causal-LM checkpoint. It does not download a model and is not
started while a training run is using the CPU. Each supplied option is scored
independently; choice, binary, and ordinal requests retain the strict typed
response format.

Once a local Qwen3-0.6B-Base checkpoint is available, evaluate the frozen
Turkish negation seed without using its labels for prompt or threshold tuning:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_v0_negation.py `
  --model-path data\raw\models\Qwen-Qwen3-0.6B-Base `
  --output-dir results\qwen3_0_6b_v0_negation
```

## V1 Turkish pilot data

Generate a small, explicitly synthetic decision-training pilot from MASSIVE
train/dev. It produces paired answerable and insufficient-option records and
never reads the MASSIVE test split or frozen ReflexBench-TR:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_v1_massive_pilot.py `
  --output data\prepared\v1\massive_tr_pilot.jsonl `
  --train-limit 600 --validation-limit 120
```

The pilot is for testing the V1 training path; it is not a native Turkish
benchmark or an unseen-option generalization result.

After a V1 training run, `scripts/run_v1_decision.py` loads its saved
`backbone/`, `decision_heads.pt`, and validation calibration metadata. It
accepts the same typed request JSON as the V0 runner and returns a bounded
decision, `P(answerable)`, and separate fallback metadata.

The V1 trainer saves the best epoch before final calibration and artifact
writing. If that final stage fails, rerun it with the original training
arguments and `--finalize-checkpoint results\<run>\checkpoints\best.pt`;
it verifies the data hash and run configuration and does not repeat training.
For an independent frozen test file, use
`scripts/evaluate_v1_decision_model.py --model-dir results\<run> --data
<test.jsonl> --output-dir results\<test-run>`. The evaluator accepts only
`split=test` records, rejects train/validation state overlap, and never
refits calibration on test data.

## Native Turkish benchmark annotation

`benchmarks/reflexbench_tr/annotation_templates/items.tsv` is a blank
collection template for independent native-Turkish annotators (see
`ANNOTATION.md`). Each row moves through `blank -> drafted ->
double_annotated -> adjudicated`; only an adjudicator may set
`adjudicated_label`, and only after both `annotation_1` and `annotation_2`
are recorded together with their annotator ids, ISO-8601 dates, and
guideline version -- with no dangling audit field on a step that was never
filled in. Every family carries structured `options_json` (`{"option_id":
"option text"}`, at least two), because that is the actual runtime input the
V1 decision path and the answerability head consume, not a free-text label.
Every label must name a supplied option key, with one exception:
`answerability` items may also be labeled `insufficient`, meaning none of
the supplied options is answerable from the state.

```powershell
.\.venv\Scripts\python.exe scripts\lint_annotation_template.py `
  benchmarks\reflexbench_tr\annotation_templates\items.tsv
```

`lint_annotation_template.py` rejects inconsistent rows (a label set out of
order, a label that names no supplied option, an unknown `task_family`, a
duplicate `item_id`, a missing or non-ISO-8601 annotator/adjudicator date, an
`answerability` item that defines an option literally named `insufficient`)
and reports how many items in each task family have reached each stage. Once
both annotators have recorded labels, compute raw agreement and Cohen's
kappa per task family (kappa is reported as undefined, not 1.0, when both
raters only ever used one category), and ingest only adjudicated rows into a
frozen JSONL file:

```powershell
.\.venv\Scripts\python.exe scripts\adjudicate_annotation_template.py `
  benchmarks\reflexbench_tr\annotation_templates\items.tsv `
  --ingest benchmarks\reflexbench_tr\expansion_frozen.jsonl
```

`--ingest` refuses to overwrite an existing frozen file; choose a new path
per batch. Every family is ingested in the same
`options`/`correct_option`/`answerable`/`split: "test"` contract as
`V1_DATA.md`, so the output is directly usable by
`scripts/validate_v1_decision_data.py` and
`scripts/evaluate_v1_decision_model.py` with no conversion step. An
`answerability` item adjudicated `insufficient` is emitted with
`answerable: false, correct_option: null`; every other item (including
`relevance`, now a `relevant`/`not_relevant`-style choice) is emitted with
`answerable: true` and `correct_option` set to the adjudicated option. Every
record includes a `provenance` block (annotator/adjudicator ids, dates,
guideline version, and the row's `notes` as adjudication rationale) for
auditability. Neither tool invents item text, translates an English seed, or
infers a label; both only check structure and aggregate what annotators
already wrote.

## Release checklist: results registry

Before a release, verify every RESULTS.md row that claims a completed run
links to a real, parseable result file, and that no completed result on disk
was left out of RESULTS.md:

```powershell
.\.venv\Scripts\python.exe scripts\validate_results_registry.py
```

It exits non-zero and lists the gap if a link is missing/broken or an
artifact under `results/` (other than a `*smoke*`/`*_eval` gate) has no
RESULTS.md row.

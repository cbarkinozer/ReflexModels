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

## V0 causal-LM option scoring

`run_v0_causal_choice.py` runs the non-generative duplicated-branch oracle
against a local causal-LM checkpoint. It does not download a model and is not
started while a training run is using the CPU. Each supplied option is scored
independently; choice, binary, and ordinal requests retain the strict typed
response format.

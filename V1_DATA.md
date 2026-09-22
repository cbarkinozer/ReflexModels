# V1 decision-training data contract

V1 adds an explicit answerability target. A decision record distinguishes
"one supplied option is supported" from "the state is insufficient or none of
the options is appropriate." A restricted option softmax alone cannot learn
that distinction.

```json
{
  "id": "tr-example-001",
  "language": "tr",
  "origin": "native",
  "source": "curated-v1",
  "split": "train",
  "state": "User wants to inspect only local project files.",
  "question": "Which tool should be used?",
  "options": {"filesystem": "Read local files", "web": "Search the internet"},
  "answerable": true,
  "correct_option": "filesystem"
}
```

Actual Turkish-first records must use `language: "tr"` and preserve UTF-8
text. For an insufficient-evidence example, `answerable` is `false` and
`correct_option` is `null`. `origin` is provenance (`native`, `translated`,
or `synthetic`); `source` identifies its registered source; and `split` is
`train` or `validation`. Threshold selection uses validation data only;
frozen ReflexBench-TR remains evaluation-only.

The first reproducible Turkish pilot is generated with
`scripts/prepare_v1_massive_pilot.py` from MASSIVE train/dev only. It samples
source utterances across intent labels and creates one answerable and one
insufficient-option record for each utterance. The two records have the same
state and question but different option sets, so a model must inspect the
supplied options to predict answerability. These derived records are marked
`origin: "synthetic"` and retain the source utterance ID and its original
`translated` provenance. They are training-path data, not native benchmark
data or an unseen-option generalization test. The Turkish intent descriptions
in `configs/massive_intent_descriptions_tr.json` are project-authored pilot
phrases and should be reviewed before a release dataset is assembled.

The V1 training runner scores options in isolated branches. Its answerability
head receives the shared state/question representation plus order-invariant
mean and max summaries of the option-branch representations. This lets the
same state/question have different answerability targets when the supplied
options differ. Improved validation checkpoints are written to
`<output-dir>/checkpoints/best.pt` before later epochs continue. If final
calibration or artifact writing fails, rerun the trainer with the same model,
data, and hyperparameter arguments plus `--finalize-checkpoint
<output-dir>/checkpoints/best.pt`. It verifies the checkpoint's original
configuration and training-data hash, then skips all training epochs.
Metrics computed on the same validation split used for checkpoint selection,
temperature fitting, and threshold selection are development diagnostics, not
unbiased final performance estimates. Final claims require a separate frozen
evaluation set. `scripts/evaluate_v1_decision_model.py` accepts only
`split="test"` JSONL, uses the saved temperature and threshold without
refitting them, reports Turkish and English separately, and rejects normalized
state-text overlap with the model's recorded train/validation states. It writes
machine-readable metrics and per-record predictions for audit. Keep the test
set frozen before inspecting its results; repeated tuning against it would
still contaminate the estimate even when the script itself does not train.

`--freeze-backbone` is an optional low-cost pilot mode that updates only the
two decision heads. It is a diagnostic baseline, not the full V1 fine-tuning
objective. The run metadata records whether it was used.

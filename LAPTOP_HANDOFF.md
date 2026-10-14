> Current continuation (2026-10-15): annotation, training and fresh evaluation are complete. Continue desktop A32 testing from [DESKTOP_HANDOFF_20261015.md](experiments/decision-followups/DESKTOP_HANDOFF_20261015.md) and the current [TODO.md](experiments/decision-followups/TODO.md). The laptop instructions below are historical.

# Continue phone-routing review on the laptop

Current stage (2026-10-12): **human data review, before any new GPU training**.
We plan three independent reflex-1 adaptations: phone-only, general-only and
combined. Models remain local/unpublished. Do not rent/train yet.

The 1,080-row CSV is at
`experiments/decision-followups/phone-routing-review-sheet.csv`: 900 training,
90 calibration and 90 historical test rows. Only review UI v3 is current.
Route and review decision are required; scenario categories are optional prepared
dropdowns. `other` means a missing category; `unclear` means ambiguous wording.
Offline availability has its own category. Label clear requests by intended
route even where the current app tool would need future implementation work.

**Laptop agent task:** start a local Argilla instance, import the CSV, restore
the committed annotation snapshot, and give the user the new laptop-local train
review link. Follow
[review-ui/README.md](experiments/decision-followups/review-ui/README.md).
Do not reuse this Mac's dataset UUID links: a fresh server generates new IDs.
The snapshot preserves submitted training reviews. Do not overwrite
reviews or silently relabel human decisions. Ask the user to click Submit for
each reviewed row, then export their progress back into the snapshot.

General data is prepared and CPU-checked; corpus files/model weights are ignored
and not transferred by Git. Full source commits, split IDs/hashes and reproduction
instructions are in `experiments/decision-followups/GENERAL_DATA.md`. Training
scripts and shared helpers are committed in the companion pocket-agents-lab repo.

Still pending: finish human review; resolve ambiguous examples and
paraphrase groups; seal a genuinely new phone test cohort; obtain current GPU
SSH/rates; smoke all three pipelines; train/evaluate/export; benchmark A32;
implement one-backbone decision/text app integration. Existing phone test results
are historical, not fresh evidence. W&B is only for actual fine-tuning.

Read `experiments/decision-followups/TODO.md`, `PLAN.md`, `ARGILLA_REVIEW.md` and
`AGENTS.md` for details. Preserve frozen source files; reviewed corrections must
be applied to a new dataset version. Save configs/logs/costs and verify model
backups before closing any future pod.

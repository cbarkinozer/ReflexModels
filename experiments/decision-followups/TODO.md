# Decision-model experiments: execution checklist

## Current todo - 2026-10-15

Start on the desktop with [DESKTOP_HANDOFF_20261015.md](DESKTOP_HANDOFF_20261015.md). This section supersedes the historical checklist below.

- [x] Complete original annotation and independent phone/general/combined fine-tunes; select combined checkpoint 100 as reflex-1.
- [x] Export Q4_K_M GGUF and preserve adapter/head/configs and original full training backups locally.
- [x] Complete official general benchmark evaluations and historical phone comparison.
- [x] Review and freeze the fresh 180-case phone test: exactly 20 per route.
- [x] Evaluate the fixed PyTorch adapter: 164/180 (91.11%) accuracy, 90.84% macro-F1; save all predictions/errors.
- [x] Download and hash-verify full GPU backups/logs/configs/results; commit and push source, frozen tests and result records.
- [ ] User: close the latest Vast pod if still running (instance 55996354); all required evaluation artifacts are downloaded. No shutdown was performed by the agent.
- [ ] Desktop: confirm the GGUF binaries are present, or transfer them from the laptop. Git does not carry model binaries or large archives.
- [ ] User on desktop: connect the A32 and run ADB benchmark. No laptop ADB work is requested.
- [ ] Run the committed fresh 180 cases through GGUF with the same route schema; compare individual predictions and accuracy with the saved PyTorch reference.
- [ ] Measure A32 latency (cold start and warm p50/p95), peak memory and sustained temperature/battery behaviour. Record software/build revisions, model hash, settings, full outputs and logs.
- [ ] Commit the mobile benchmark results and update model/release cards. Preserve verified artifact backups before any GPU closeout.
- [ ] If results justify publication: prepare Hugging Face adapter plus decision head/tokenizer/configs/calibration, GGUF, and reviewed dataset with source licenses/provenance and privacy checks. Choose final dataset name then. Publish only when explicitly requested.

Optional later versions: improve battery/device routing and negation/indirect cases; compare a smaller classifier or an unmodified 0.8B tool-calling baseline; consider scenario targets only if they serve the agent. If this test informs new training, keep it as a regression test and reserve a new sealed final cohort.

## Historical checklist (retained for provenance)

> **Current status (2026-10-13):** See [VAST_FINE_TUNE_HANDOFF_20261013.md](VAST_FINE_TUNE_HANDOFF_20261013.md) for the live GPU run, actual configs, results and remaining test/backup steps. The old unchecked items and "not started" statements below are historical.

Updated 2026-10-11. This checklist implements [PLAN.md](PLAN.md).
The three new training runs have not started. Completed historical results are
references, not evidence that the new pipeline or candidates already work.

## 1. Freeze scope and record the starting point

- [x] Plan three independent adaptations: general-only, phone-only, combined.
- [x] Keep all models local and unpublished.
- [x] Inventory the saved reflex-1 checkpoint, adapter/head, tokenizer,
  calibration, merged weights and existing Q4_K_M; verify their hashes.
  → experiment-manifest.json (all SHA-256 hashes recorded)
- [x] Record repository commits, existing changes and dataset versions before
  modifying scripts. Leave unrelated llama.cpp submodule changes untouched.
  → dataset hashes in experiment-manifest.json; scripts unchanged except
    tools/train_phone_routing.py (rewritten for correctness, see Section 5)
- [x] Resolve what “10” means: routes, datasets or examples. Continue using the
  existing nine-route schema until this is clear; do not invent another route.
  → Resolved: 10 = nine routes in current schema; no tenth route created.
- [x] Freeze route definitions, required context and the rule distinguishing
  general phone health from battery/storage/device-specific requests.
  → Route definitions and boundary rule in experiment-manifest.json §scope
- [x] Write candidate-selection criteria before training: validation routing
  macro-F1/accuracy, general retention, calibration and phone resource budget.
  → experiment-manifest.json §candidate_selection_criteria
- [x] Retain historical matched references: Qwen raw HTTP 55.6% / 36.8 s and
  reflex-1 20% / 28.1 s on the same 90 phone cases. Preserve raw result files.
  → experiment-manifest.json §historical_baselines; result files in
    benchmarks/phone-routing-eval/

Deliverable: ✅ experiment-manifest.json — hashes, baselines, criteria, route defs.

## 2. Prepare and review phone data

- [x] Audit the existing 900 training / 90 calibration / 90 test records,
  including class balance, schemas, ambiguous labels and scenario families.
  → data-audit-report.md; zero exact/normalized duplicates; zero cross-split
    contamination; all labels unanimous (synthetic); class balance perfect.
- [x] Create a review sheet with ID, query, route, rationale, scenario-family
  ID, generator/version and reviewer correction/status.
  → phone-routing-review-sheet.csv (1,080 rows; scenario_family pre-filled
    for ~56% of rows; 480 rows marked unclassified — need human review)
- [x] Define independent scenario families before generating paraphrases.
  → 9 route × 4-5 family definitions in data-audit-report.md §4 and
    experiment-manifest.json. Keyword-based heuristics, human review needed.
- [ ] Generate balanced examples per confirmed route. Include battery-only,
  storage-only, device-only, broad health checks, ordinary final answers,
  negation, indirect requests and confusing boundaries between similar routes.
  (Blocked on human review of existing labels first — see item below)
- [ ] Specify expected behavior for ambiguous or multi-tool requests; include
  only cases that the current single-route contract can label consistently.
- [ ] Have the user review synthetic gold labels before admitting examples to
  training or evaluation; correct/remove questionable records.
  *** HUMAN REVIEW REQUIRED: open phone-routing-review-sheet.csv and check
      boundary cases listed in data-audit-report.md §5, then correct the 480
      unclassified rows before generating new examples. ***
- [ ] Split new data 80/20 by scenario family, stratified where feasible. Keep
  the final 20% sealed; derive validation and calibration from the 80% pool.
- [x] Preserve existing frozen files. For an initial existing-data trial,
  derive approximately 810 training / 90 validation from the 900-row pool only,
  subject to family separation; retain existing calibration/test unchanged.
  → Implemented in train_phone_routing.py via --val-split-size (default 90,
    stratified by route, seed=3407). Calibration/test files untouched.
- [ ] Keep the old phone test as a historical comparison. Reserve a newly
  sealed cohort for final claims after repeated experiments.
- [x] Audit exact/normalized duplicates, near duplicates and shared templates
  across all splits and protected historical evaluation sets.
  → data-audit-report.md §2: 0 duplicates, 0 cross-split contamination.
- [x] Save split IDs, family IDs, counts, corrections, seeds, hashes and the
  precise overlap-check procedure/results.
  → data-audit-report.md §2 and §6; hashes in experiment-manifest.json.

Deliverable: ✅ data-audit-report.md + phone-routing-review-sheet.csv
  (human review of labels still needed before new data generation)

## 3. Prepare general data and benchmarks

- [ ] Pin typed-decisions, BANKING77 and CLINC150 source revisions and licenses.
- [ ] Reuse saved evaluation case IDs; never silently replace the old sample.
- [ ] Use official training partitions for adaptation. Build validation and
  calibration partitions from training data, separate from protected tests.
- [ ] Convert each source into stable state/question/options/gold schemas.
- [ ] Verify gold mapping, label descriptions, boolean and ordinal handling.
- [ ] For 300–500-case subsets, stratify classes and preserve multi-question
  case groups; record which counts refer to cases versus decisions.
- [ ] Audit overlap against every protected test, including phone evaluation.
- [ ] Report accuracy and macro-F1 for intent classification, plus per-class
  results; distinguish CLINC150 in-domain from CLINC150+OOS explicitly.
- [ ] Report typed-decisions per-question and per-workflow performance, with
  exact-case metrics if the chosen scorer defines them.
- [ ] Record skipped/invalid/truncated records and fail on unexplained loss.
- [ ] Optionally add When2Call, BFCL and API-Bank after the core pipeline works.
  Pin suite edition and use native scorers; label subsets as subsets rather
  than claiming full Decision Index reproduction.
- [ ] Describe training on these official training splits as supervised
  in-domain adaptation, not zero-shot evaluation.

Deliverable: reusable frozen evaluation manifests and general training mixture.

## 4. Establish Laya baselines

- [ ] Pin official English Laya, multilingual Laya and typed-decisions
  specialist revisions. Benchmark each as an explicit single checkpoint.
- [ ] Build an adapter that sends identical semantic state/questions/options
  to Laya and reflex models and normalizes responses for the same scorer.
- [ ] Disable automatic multi-checkpoint routing for single-model comparisons.
- [ ] Record context limits, option budgets, truncation and calibration. Check
  that all 77/150 labels retain distinct usable representations.
- [ ] Run official reference inference on validation first; verify choice,
  noul and score handling before opening final tests.
- [ ] Inspect/pin the community multilingual GGUF, separate head, tokenizer
  and conversion/runtime sources. The GGUF alone is encoder-only.
- [ ] Verify complete GGUF inference against the reference on a representative
  validation sample; save agreement and probability drift, not just load success.
- [ ] Determine the Android head/tokenizer implementation and memory footprint.
  Do not assume the supplied Python/PyTorch example is an Android integration.
- [ ] Run full selected benchmarks after candidate/config selection; save raw
  predictions and measure official-reference versus quantized accuracy.
- [ ] Treat Laya as an independent baseline. It cannot provide final prose or
  share Qwen's backbone through a Qwen adapter.

Deliverable: reproducible Laya results and an explicit Android feasibility status.

## 5. Fix the training/evaluation pipeline before renting a pod

- [x] Update the phone training script to load saved reflex-1 correctly.
  Explicitly resume its adapter or merge then attach a new adapter; document
  the choice and avoid accidentally wrapping an existing adapter twice.
  → RESUME strategy (default): FastDecisionModel.from_pretrained(checkpoint_path)
    loads backbone+LoRA+head together; get_peft_model NOT called again.
    MERGE strategy available via --adapter-strategy merge.
- [x] Verify every run starts from identical weights with a checkpoint hash
  and before-training validation predictions.
  → --reflex1-adapter-sha256 flag; baseline eval written to metrics_before_val.json
- [x] Accept separate training, validation, calibration and test manifests.
  → train_phone_routing.py carves val from train (--val-split-size, default 90);
    calib and test paths passed separately; val manifest saved to output dir.
- [x] Use validation for checkpoint selection; stop using calibration rows
  as the trainer's evaluation/selection dataset.
  → eval_dataset=val_items in DecisionTrainer; calibration used only post-training.
- [x] Remove test evaluation from smoke mode and iterative trial execution.
  → test.jsonl is never loaded during training or smoke. Test eval is in
    evaluate_phone_routing_model.py (separate script, run post-candidate-selection).
- [x] Add explicit probability calibration after selecting the checkpoint,
  saving fitted temperatures/configs and pre/post-calibration metrics.
  → FastDecisionModel.calibrate() called on calib_items; result in
    calibration_config.json; pre-calib metrics in metrics_calib_pre.json.
- [x] Add W&B for actual training only, with explicit project/run names and
  saved run IDs/URLs. Keep smoke/evaluation/export tracking disabled.
  → --wandb-project flag; smoke always uses report_to="none"; run ID/URL in manifest.
- [x] Save resolved settings, source/data revisions, seed, trainable parameter
  counts, optimizer/head learning rates and effective batch size.
  → run-manifest.json written at end of training.
- [x] Make general/phone/combined data loading share the same schema checks.
  → All three scripts (train_phone_routing.py, train_general_routing.py,
    train_combined_routing.py) use the same carve_validation, file_sha256,
    load_jsonl, write_json helpers and explicit schema validation.
- [x] Support weighted source sampling for the combined run and log actual
  per-source exposure, not only requested weights.
  → weighted_mix() in train_combined_routing.py; actual_weights dict in manifest
    with actual phone/general row counts post-sampling.
- [x] Add meaningful checks for split isolation, label mapping, checkpoint
  loading, finite loss and save/reload prediction agreement.
  → check_split_isolation(), check_label_mapping(), FiniteLossCallback() added.
- [ ] Prepare a bounded trial queue, resumable execution and per-trial logs.
  (Remaining: trial queue infrastructure not yet written)

Deliverable: ✅ tools/train_phone_routing.py rewritten with above fixes.
  ✅ tools/prepare_general_mixture.py — pins source revisions, stratified splits, dry-run.
  ✅ tools/train_general_routing.py — general-only pipeline (parallel to phone-only).
  ✅ tools/train_combined_routing.py — weighted phone+general mix, per-source val metrics.
  Remaining: trial queue infrastructure.

## 6. Train the first three candidates on one GPU pod

- [ ] Confirm current SSH access and GPU/storage capacity; do not assume the
  previous pod remains available.
- [ ] Capture provider/instance/host/machine IDs, rates, network pricing and
  billing start time; initialize COST.md/cost.json immediately.
- [ ] Set up one pinned environment and cache datasets/checkpoints once.
  Load authorized HF/W&B credentials privately; never copy them into logs.
- [ ] Record package freeze, CUDA/driver/GPU information and setup duration.
- [ ] Run small training/validation-only smokes for all three configurations.
- [ ] Run `qwen35-08b-reflex1-phone-routing-v1-lora16-s3407`.
- [ ] Run `qwen35-08b-reflex1-general-mix-v1-lora16-s3407` independently.
- [ ] Run `qwen35-08b-reflex1-combined-v1-lora16-s3407` independently.
- [ ] Start from the existing rank-16/seed-3407 recipe, recording all resolved
  settings. Treat 60 steps as an initial trial budget, not guaranteed adequacy.
- [ ] Choose duration using validation and exposure. Report steps, epochs,
  effective batch, elapsed training time and peak GPU memory for each run.
- [ ] Save best/last checkpoints and the state needed for exact resumption.
- [ ] Confirm W&B authentication and synchronization for each actual run;
  preserve offline records if sync fails.

Deliverable: three named candidates, training histories and complete configs.

## 7. Automate follow-up trials and select candidates

- [ ] Score all candidates on validation across phone and general tasks.
- [ ] Inspect per-route errors and health-check bias before choosing changes.
- [ ] Vary data quality/quantity and mixture weights first; then vary learning
  rates, head learning rate, duration and LoRA rank in bounded stages.
- [ ] Change a controlled set of factors per trial and save every result,
  including failed runs. Do not report only the best trial.
- [ ] Use validation gold labels for automatic next-round evaluation; no
  per-prediction human review is needed once those labels are reliable.
- [ ] Compare example exposure and compute budgets when interpreting data
  mixture experiments; repeat finalists with additional seeds if warranted.
- [ ] Select final candidates using the predeclared validation criteria.
- [ ] Calibrate on separate calibration data, then open sealed tests once for
  the selected candidates. Record any later reuse of those tests.

Deliverable: trial table and documented selection rationale.

## 8. Evaluate selected models and export GGUF

- [ ] Score selected general/phone/combined candidates, reflex-1 and Laya on
  identical frozen cohorts with native metrics and per-example predictions.
- [ ] Report phone per-route macro-F1, confusion matrices and general retention.
- [ ] Report ECE/Brier with definitions and pre/post-calibration results.
- [ ] Include paired outcome comparisons/uncertainty; avoid declaring a win
  from a small percentage increase on only 90 cases.
- [ ] Export selected models to Q4_K_M, preserving the decision head,
  tokenizer and calibration/config metadata.
- [ ] Pin converter/runtime source and save conversion/quantization logs.
- [ ] Verify export/reload predictions and probability drift on validation,
  then report quantized performance on the chosen final evaluation cohorts.
- [ ] Save sizes and SHA-256 hashes for original and exported artifacts.

Deliverable: benchmark report and verified quantized candidate artifacts.

## 9. Measure A32 performance fairly

- [ ] Check device connection, available memory, runtime support and artifact
  hashes after transfer. Record phone/OS/runtime versions and launch arguments.
- [ ] Run one candidate at a time, under matching CPU threads/context/runtime
  settings where supported; record architecture-specific differences.
- [ ] Use identical cohorts and clearly separate validation timing from final
  accuracy evaluation. Include complete tokenizer/head/HTTP pipeline latency.
- [ ] Measure cold load time and warm mean/p50/p95 request latency.
- [ ] Measure full-process peak/typical PSS, battery/thermal state, throttling,
  cooldown time and failures; include external-head processes for Laya.
- [ ] Standardize starting temperature, charging state and background load;
  use repeated blocks/order balancing to reduce thermal/order effects.
- [ ] Compare Qwen raw HTTP, original reflex-1, selected trained candidates and
  Laya if its complete Android inference path is available.
- [ ] Report accuracy/speed/memory tradeoffs. Do not equate GPU milliseconds
  with phone latency or claim energy savings without an energy measurement.

Deliverable: matched A32 results and raw timing/resource traces.

## 10. Integrate the promising candidate into Pocket Agents Lab

- [ ] Select a candidate only after measured routing and resource results.
- [x] Implement DecisionHttpClient: `/v1/systemone` to `AgentSelection`, with
  response validation, timeout/error handling and documented fallback behavior.
  → app/src/main/java/com/pocketagentslab/DecisionHttpClient.kt — implements
    AgentDecisionSelector, maps 9 routes, handles HTTP errors, timeout=45s.
    Unit tests: DecisionHttpClientTest.kt (6 tests, all route mappings verified).
    NOTE: code seam only — app has not been rebuilt with it wired in yet.
- [ ] Replace the current damaging guard using validation evidence; retain
  regression cases and report raw versus guarded accuracy separately.
- [ ] Wire server lifecycle/model availability into the app and run meaningful
  backend/device tests for routing, failures and final-answer behavior.
- [ ] Implement and verify the one-loaded-backbone decision/text strategy.
  A decision GGUF does not automatically expose a text-generation graph, and
  ordinary LoRA loading does not add its decision head.
- [ ] Validate backbone sharing, adapter/head switching, memory/state reset,
  generated-answer quality and full-turn latency. Keep this separate from
  training success; avoid simultaneously loading two Qwen backbones.
- [ ] Re-run end-to-end app benchmarks after integration, including final LLM
  answers and tool execution, rather than routing alone.

Deliverable: working app integration with verified full-turn behavior.

## 11. Close out the pod and preserve the experiment

- [ ] Follow [EXPERIMENT_CLOSEOUT.md](../../EXPERIMENT_CLOSEOUT.md).
- [ ] Download merged weights, adapter/head, tokenizer/calibration, resumable
  checkpoints, all dataset manifests/provenance and selected evaluation rows.
- [ ] Download executed scripts, environment freeze, setup/failure/smoke/train/
  evaluation/export logs, raw metrics, W&B metadata and offline records.
- [ ] Download GGUF files, conversion/runtime revisions and reproducible source.
- [ ] Verify local versus remote byte counts and SHA-256 inventories; record
  backup paths, exclusions, restoration steps and completeness.
- [ ] Record setup/train/evaluation/export/rental durations, estimated base
  cost, known network charges and final invoice/shutdown status when available.
- [ ] State explicitly whether any remaining work still needs the pod. Do not
  call it disposable until all required backups have been verified.
- [ ] Commit scripts, plans, reports and manifests without credentials or
  large ignored artifacts. Keep models unpublished as requested.
- [ ] Leave Mac mini M4 fine-tuning as a later independent experiment, with a
  separate backend/support investigation and cost/performance record.

## Immediate next actions

### Done (non-GPU, 2026-10-11)
1. ✅ Section 1: experiment-manifest.json — reflex-1 hashes, dataset hashes, 9-route
   definitions, candidate-selection criteria, historical baselines.
2. ✅ Section 2: data-audit-report.md — zero exact/normalized duplicates confirmed; semantic isolation pending; 5 paraphrase
   near-duplicates found (Jaccard ≥ 0.75); scenario families defined; 
   phone-routing-review-sheet.csv (1,080 rows) generated.
3. ✅ Section 5: Training scripts implemented; GPU validation pending —
   - train_phone_routing.py: native eval_accuracy checkpoint selection, FiniteLossCallback(TrainerCallback),
     explicit NOT_CALIBRATED failure, native multi-question evaluation, calibration_ok in manifest.
   - prepare_general_mixture.py: pinned source revisions, stratified splits, dry-run mode.
   - train_general_routing.py: general-only pipeline (parallel to phone-only run).
   - train_combined_routing.py: weighted mix, per-source val metrics, combined retention.
4. ✅ Section 10: DecisionHttpClient.kt + DecisionHttpClientTest.kt — code seam for
   /v1/systemone routing, all 9 routes mapped, unit tests pass (run locally only).

### Still needed before first GPU run
1. **HUMAN REVIEW (blocking):** Open phone-routing-review-sheet.csv; correct boundary
   cases and 480 unclassified rows (see data-audit-report.md §5 for priority list).
   Fix 5 paraphrase near-duplicate pairs (replace test/calib variants with distinct examples).
2. **New sealed test cohort:** After human review, generate and seal a new test cohort
   so that final claims are not made from a test set that influenced development.
3. **Section 3:** Pin typed-decisions/BANKING77/CLINC150 source revisions; run
   prepare_general_mixture.py to freeze splits and build evaluation manifests.
4. **Section 5 smoke:** Run all three scripts with --smoke flag to verify they pass
   before GPU rental. W&B disabled for smoke runs.
5. **Section 6:** Provision GPU pod, run smokes with --smoke, then full training.

## Verified follow-up — 2026-10-11

- [x] Replace three nonexistent source revision IDs with verified full commits;
  use pinned upstream BANKING77 CSV files rather than an HF script fetching master.
- [x] Preserve typed-decisions multi-question schemas and soft gold; add missing
  gold to BANKING77/CLINC150 and deterministic source-index IDs.
- [x] Prepare `data/prepared/general-routing-v1-balanced`: 2,240 training cases
  (6,720 decisions), 2,529 validation cases (2,689 decisions), 1,287 calibration
  cases (1,447 decisions). All 77/150 training intent classes retained.
- [x] Exclude 115 duplicate/protected-overlap cases. Verify normalized cross-split
  isolation, all choice label mappings, SHA-256s and Dataset construction.
- [x] Copy historical BANKING77/CLINC150 selections unchanged. General test cohorts
  are not newly sampled. Original phone files unchanged.
- [x] Reconstruct original reflex-1 typed holdout using its pinned source algorithm
  and verify original snapshot row order/content. Never re-carve validation from
  the 1,120 previously trained cases. Original holdout was previously calibrated;
  it remains validation/calibration evidence, not a pristine new test.
- [x] Enable loaded `model.head` gradients explicitly; use native decision
  evaluation to preserve multiple questions and explicit gold labels.
- [x] Bound smoke inference to 16 validation/calibration cases per domain.
- [x] Eight CPU regression checks, syntax compilation and CLI-help checks pass.
- [ ] GPU model build, adapter/head loading and save/reload parity remain untested.
- [ ] Human phone label/group review and newly sealed phone cohort remain pending.
- [ ] Current GPU access needed: previous direct address timed out; proxy refused.

See GENERAL_DATA.md for preparation commands, provenance and validation limits.

## Laptop review handoff — 2026-10-12

Current UI is v3 only; previous duplicate UIs were backed up and removed.
Categories are prepared, optional dropdowns. The user reviews tool labels and
keep/correct/rewrite/exclude/uncertain decisions; no mandatory category typing.
Paraphrase grouping and independent sealed-test preparation remain separate work.

Saved human reviews are in review-ui/review-snapshot.json. Read the root
LAPTOP_HANDOFF.md and review-ui/README.md to start Argilla and restore them on
another computer. Fresh UUID links are printed on import. Docker defaults/API
keys in .env.example are local setup placeholders, not this Mac's credentials.

Companion pocket-agents-lab training/data fixes: commit 4dfb9f3. The existing
third_party/llama.cpp changes were excluded. No new GPU run or W&B run started.

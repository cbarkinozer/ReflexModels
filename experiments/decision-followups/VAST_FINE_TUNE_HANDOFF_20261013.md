> Current status ? 2026-10-14: training and GGUF export completed previously; the release README records the original training instance as destroyed. Held-out evaluation is now complete on new RTX 4090 instance `55976079` (`ssh -p 12238 root@ssh4.vast.ai`), with a SHA-256 verified local results archive. The new instance remains running at approximately $0.53037/hour. See [held-out results](HELDOUT_EVALUATION_20261013.md). The historical execution checklist below predates this completed evaluation.

# Decision fine-tunes: live handoff (2026-10-13)

**Latest user direction:** download and verify all artifacts, but **do not destroy the Vast instance**. Leave it running for the user after the work; the hourly rental continues until they stop/destroy it themselves or give a new instruction.

This is the current operational record for continuing the GPU work in Claude Code or another terminal. The older `PLAN.md` and `TODO.md` describe an earlier checkpoint-based plan and have stale status. The user clarified that these three new runs should each start independently from **base `unsloth/Qwen3.5-0.8B`**, with no reflex-1 adapter. Do not treat any old reflex-1 result as one of the new fine-tunes.

The final held-out protocol/results and leakage interpretation are in [HELDOUT_EVALUATION_20261013.md](HELDOUT_EVALUATION_20261013.md).

## Access and cost

- Vast.ai instance ID `55798314`, host `34031`, machine `14826`, RTX 4090 24 GB, 80 GB disk.
- SSH: `ssh -p 23910 root@ssh9.vast.ai` (the earlier IP endpoint did not work).
- Remote project: `/workspace/reflex-decision`.
- Vast instance API snapshot: `dph_total = $0.4429609437108856/hour`, `dph_base = $0.413331314081256/hour`, start epoch `1791919124.3796685`. Estimate compute + disk cost as elapsed running hours × `dph_total`; network/other charges and final invoice can differ. Refresh rate and elapsed time before closeout with `vastai show instance 55798314 --raw` (CLI available on pod) or the Vast dashboard. **The pod continues to bill while running.**
- At `2026-10-13 20:19:50 UTC`, elapsed time was about `1.0182 h`, giving a provisional `$0.4510` at the reported hourly rate. This is a point-in-time estimate, not the final charge.
- Later snapshot (`vastai show instance 55798314 --raw`, taken ~`2026-10-13 23:40 UTC`): `duration = 5983.95 s` (`1.6622 h`), compute cost ≈ `1.6622 × $0.44296 ≈ $0.7363`, plus `inet_down_cost ≈ $0.0260` and `inet_up_cost ≈ $0.0260` ⇒ provisional total ≈ **`$0.79`**. Instance was still `running`/billing at that point. Re-check with the same command before final stop/destroy for the true final charge.
- W&B project `c-barkinozer/reflexmodels`. The API key was copied from the local `llm-bender-lab/.env` to remote `/workspace/reflex-decision/.env` (mode 600). Never print or archive this file. Remove the transient remote `.env` after all W&B runs have synced and artifacts are backed up.

## Data and separation

- Local input bundle: `C:/Users/barkin/Documents/GitHub/pocket-agents-lab/runs/decision-followups-full-gpu-inputs-20261013.zip`; SHA-256 `0549f580f769067a4a018aed5a39f35040682af28b5e663e190290579102e3e8`. Extracted on pod.
- Phone reviewed v2: `data/prepared/phone-routing-human-reviewed-v2-1080-fixedval-20261013`: 781 train, 90 validation, 90 calibration, 88 usable historical test. Validation is frozen, separate from calibration and test. Template groups were isolated in the preparation pass. The historical phone test had been used in prior project work, so it is a comparison set, not a pristine new blind test.
- General: `data/prepared/general-routing-v1-balanced`: 2240 train, 2529 validation, 1287 calibration. Training mixture: 1120 typed decisions, 560 BANKING77, 560 CLINC150. General source revisions and construction are recorded in prep manifests/scripts.
- Combined training: 781 phone + 781 class-preserving general samples (391 typed, 195 BANKING77, 195 CLINC150); equal domain weights. All 77 BANKING and 150 CLINC classes remain represented. Combined validation has 90 phone + 2529 general cases. Checkpoint selector uses the mean of *domain accuracies*, preventing the larger general validation set from dominating.
- Normalized exact overlap checks among current train/evaluation files passed. This does not prove semantic independence or rule out pretraining exposure to public benchmarks. General results are supervised in-domain adaptation, not zero-shot.
- **Phone leakage audit after user concern:** the local script `pocket-agents-lab/runs/decision-followups-vast-20261013/audit_phone_split_similarity.py` and its JSON report found zero normalized exact train/validation pairs, two high lexical similarity pairs (`battery temperature safe`/`normal`; `storage stats`/`storage status`), and 11/90 validation requests whose nearest training request had lexical score ≥0.8. Automatic same-route template groups do not overlap according to the export audit. These checks cannot rule out semantic template reuse. The 90 validation IDs were fixed from an earlier pilot and reused for dataset comparisons, so the 98.9% phone validation score is **development-set performance**, potentially optimistic from adaptive reuse. The historical phone test is also previously exposed; collect a freshly authored blind cohort for a strong generalization claim.
- **Do not run the final test preparation/evaluator until all three training runs and validation-only hyperparameter choices are finished. Never use test metrics to select checkpoints or tune hyperparameters.**

## Environment and implementation

- Remote virtualenv `.venv`. Setup log `setup-pod.log`; dependency freeze `environment-freeze.txt`. Torch `2.8.0+cu128`, Transformers `5.2.0`, TRL `0.22.2`, W&B `0.25.0`, pinned Unsloth git revision in freeze/log.
- `tools/train_general_routing.py` uses `--max-seq-length 4096`: 2048 dropped all 560 CLINC rows because its complete schema requires about 3042 tokens. The 4096 smoke and full run report zero skipped rows.
- `tools/train_combined_routing.py` now reads explicit validation files, asserts ID isolation, samples the general part while retaining all intent classes, and selects on equal phone/general domain accuracy. The output configs/manifests are in each run directory.
- `tools/prepare_followup_test_sets.py` and `tools/evaluate_decision_followups.py` are already on the pod; local source is under `pocket-agents-lab/tools`. Python compilation and data preparation checks passed, but the new evaluator has not yet been exercised on final test sets.
- All training uses LoRA rank 16, alpha 16, LR `2e-4`, seed 3407, effective batch size 32, W&B enabled. Each run begins from base model independently. Final test examples and labels were not used for training, validation, calibration, checkpoint selection or hyperparameter choice. The training manifest records the phone test file's SHA-256 fingerprint for provenance.
- Each model reports `38,147,588` trainable parameters. Optimizer training time in manifests: phone `59.5 s`, general `721.8 s`, combined 80-step `685.3 s`. Total job wall time is longer because full validation, calibration, setup and downloads add substantial time; the hourly rental covers all of it.

## Actual run commands and status

The shell exports variables from `.env` before running the following. Logs are in the project root; checkpoints, configs, final adapters and heads are in the named run dirs.

### Phone: complete

```bash
.venv/bin/python tools/train_phone_routing.py --data-dir data/prepared/phone-routing-human-reviewed-v2-1080-fixedval-20261013 --output-dir runs/phone-reviewed-scratch-full-s3407 --max-steps 60 --batch-size 8 --grad-accum 4 --lr 2e-4 --lora-r 16 --lora-alpha 16 --seed 3407 --wandb-project reflexmodels --wandb-run-name qwen35-08b-phone-reviewed-scratch-s3407
```

Log `phone-reviewed-scratch-full.log`; [W&B run](https://wandb.ai/c-barkinozer/reflexmodels/runs/5qjlvdbq). Validation `89/90 = 0.9889`; calibration `88/90 = 0.9778`; calibration fitted. No test used. Adapter is about 42 MB, decision head about 53 MB.

### General: complete

```bash
.venv/bin/python tools/train_general_routing.py --data-dir data/prepared/general-routing-v1-balanced --output-dir runs/general-balanced-scratch-full-s3407 --max-seq-length 4096 --max-steps 80 --batch-size 2 --grad-accum 16 --lr 2e-4 --lora-r 16 --lora-alpha 16 --seed 3407 --wandb-project reflexmodels --wandb-run-name qwen35-08b-general-balanced-scratch-s3407
```

Log `general-balanced-scratch-full.log`; [W&B run](https://wandb.ai/c-barkinozer/reflexmodels/runs/dpv1tutp). Zero skipped. Full validation accuracy by step: baseline `.0896`, 20 `.6508`, 40 `.7311`, 60 `.7616`, 80 `.7657`; after best-checkpoint reload `.76385` (2689 decisions for 2529 cases). Calibration fitted. No test used. This curve suggests a **single validation-only longer-step trial may be useful** if time/cost allow. A broad sweep is not justified by the phone near-ceiling validation. Preserve sealed tests until the decision is made.

### Combined: complete

```bash
.venv/bin/python tools/train_combined_routing.py --phone-data-dir data/prepared/phone-routing-human-reviewed-v2-1080-fixedval-20261013 --general-data-dir data/prepared/general-routing-v1-balanced --output-dir runs/combined-balanced-scratch-full-s3407 --max-seq-length 4096 --phone-weight 1.0 --general-weight 1.0 --max-steps 80 --batch-size 2 --grad-accum 16 --lr 2e-4 --lora-r 16 --lora-alpha 16 --seed 3407 --wandb-project reflexmodels --wandb-run-name qwen35-08b-combined-balanced-scratch-s3407
```

Original PID `7792` has exited; log `combined-balanced-scratch-full.log`; [W&B run](https://wandb.ai/c-barkinozer/reflexmodels/runs/34e5qui4). Zero skipped. Baseline phone/general `.1222/.0889`; step 20 `.8889/.6218`; step 40 `.9778/.7122`; step 60 `.9889/.7300`; step 80 `.9889/.7445`. Equal domain mean at step 80 `.8667`.

Saved best-checkpoint reload validation (`metrics_after_val.json`): phone `.98889`, general `.74489`, count-weighted combined `.75315`; calibration/ECE values are in the same file. The equal-domain mean is about `.86689`, matching the step-80 selector within evaluation precision.
Its run manifest reports `run_completed: true` and `calibration_ok: true`; the log reports W&B synced.

### Focused validation-only follow-up: complete — step 100 selected

Because combined general validation was still increasing from step 60 to 80, **one** independent 120-step combined trial was launched after the 80-step process exited, changing only `--max-steps` from 80 to 120. Launcher: `pocket-agents-lab/runs/decision-followups-vast-20261013/launch_combined_120.sh` (also copied to remote project). Remote PID `8770` at launch (exited on completion); log `combined-balanced-scratch-120step.log`; output `runs/combined-balanced-scratch-120step-s3407`; [W&B run](https://wandb.ai/c-barkinozer/reflexmodels/runs/66wubl55).

Full per-checkpoint validation (phone / general / equal mean):

| Step | Phone val | General val | Equal mean |
| --- | ---: | ---: | ---: |
| 20 | .9111 | .6210 | .7661 |
| 40 | .9667 | .7081 | .8374 |
| 60 | .9667 | .7319 | .8493 |
| 80 | .9889 | .7735 | .8812 |
| **100** | **.9889** | **.7839** | **.8864 ← best** |
| 120 | .9889 | .7780 | .8834 |

**Selected checkpoint: `checkpoints/checkpoint-100`.** It beats both the 80-step original run (equal mean `.8667`) and step 120 of this same run. The best-checkpoint reload saved as `runs/combined-balanced-scratch-120step-s3407/adapter` is the step-100 weights: `metrics_after_val.json` reports phone `.98889`, general `.78356`, count-weighted combined `.78949`, record accuracy `.79572`. Calibration fitted and OK (`calibration_ok: true`; post-calibration ECE `.0377` vs pre-calibration `.0511`). `run-manifest.json` reports `run_completed: true`. Training wall time for this run (optimizer time only, per manifest): `1052.1 s` (~17.5 min); full job including setup/validation/calibration took longer per the hourly rental.

**This is the final selected combined candidate for held-out testing: `runs/combined-balanced-scratch-120step-s3407/adapter`.**

## Remaining execution

1. ~~Wait for the running 120-step combined follow-up~~ — **done**, step 100 selected per above. The final tests remain sealed; no test evaluation has been run yet.
2. Prepare the held-out official test subsets *after* that choice:

```bash
cd /workspace/reflex-decision
.venv/bin/python tools/prepare_followup_test_sets.py --general-train data/prepared/general-routing-v1-balanced/train.jsonl --phone-train data/prepared/phone-routing-human-reviewed-v2-1080-fixedval-20261013/train.jsonl --output-dir data/prepared/followup-official-test-s3407
```

Expected output: typed 400, BANKING77 462 (6/class), CLINC150 450 (3/class), CLINC OOS excluded. The script asserts no exact normalized train/test overlap and writes a manifest. Investigate any assertion failure; do not weaken the exclusion.

3. Run the final evaluator for **each saved adapter** (and ideally the untouched base) on the *same* historical phone test and official general subsets, e.g.:

```bash
.venv/bin/python tools/evaluate_decision_followups.py --model runs/phone-reviewed-scratch-full-s3407/adapter --phone-test data/prepared/phone-routing-human-reviewed-v2-1080-fixedval-20261013/test.jsonl --general-test-dir data/prepared/followup-official-test-s3407 --output-dir runs/eval-phone-scratch --max-seq-length 4096 --batch-size 8
```

Confirm argument names/model path from `--help` and actual saved run dir before executing. Repeat with general and combined adapters and base ID `unsloth/Qwen3.5-0.8B`. Require zero unexplained skipped cases; retain per-source metric/build-report JSON and summary JSON. The current evaluator does **not** write per-case predictions. Compare per-source metrics and calibration, plus caveats above. Evaluator loading the saved adapter also tests checkpoint reload.

The ready-to-run remote helper `run_heldout_evals.sh` performs test preparation and the four sequential evaluations in one pass. Call it only **after validation selection** and pass the selected adapter path, e.g. `bash run_heldout_evals.sh runs/combined-balanced-scratch-120step-s3407/adapter`. It writes a separate log and result directory for base, phone, general and selected combined. It exits on the first failure, so inspect logs before resuming manually. The helper was copied to the pod and passed `bash -n`.

4. Back up **all logs and configuration/artifacts** locally: setup log, environment freeze, smoke logs, all four full-run logs, all `runs/` dirs including checkpoints/adapters/heads and evaluations, W&B run files, prepared data/manifests, and executed scripts. The ready-to-run `archive_all_artifacts.sh` scans for the copied W&B secret and creates a project archive excluding `.env`, `.venv` and disposable caches, plus a separate Hugging Face hub cache archive containing the base-model/data downloads. Transfer both with `scp` and verify SHA-256 on both sides. Local destination: `pocket-agents-lab/runs/decision-followups-vast-20261013/`. Confirm W&B logs/artifacts are included and all runs synced. Record final instance rate, elapsed billable time, estimated cost and any extra charge visible in Vast dashboard. **Keep the instance running per the latest user instruction.**
5. Update this handoff with completed results, archive location/hash, and pod status. If continuing work on A32/Q4 export, the older plan covers that separate phase; GPU fine-tune/test completion does not imply on-device benchmarking is done.

## Quick local paths

- `C:/Users/barkin/Documents/GitHub/ReflexModels/experiments/decision-followups/` — this handoff and prior planning/audit.
- `C:/Users/barkin/Documents/GitHub/pocket-agents-lab/tools/` — source scripts.
- `C:/Users/barkin/Documents/GitHub/pocket-agents-lab/runs/` — input bundle and intended local backups.
- `C:/Users/barkin/Documents/GitHub/llm-bender-lab/.env` — credential source; do not copy into artifacts or chat.

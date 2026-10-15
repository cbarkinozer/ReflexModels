# Desktop continuation — 2026-10-15

Current source: `73b305b`. No push or publication authorized. No GPU needed for the next phone evaluation.

## Verified locally

- ADB detects Samsung SM-A325F, serial `R68T800LZ0H` (connection briefly dropped during initial inspection).
- This Mac has only the original beta GGUF under `experiments/unsloth-qwen35-08b-decision/runs/pod-backup/reflex-1-gguf/`. The selected combined release binaries are absent from this Mac's ReflexModels tree and Downloads.
- Phone has `/sdcard/Download/models/reflex-1-Q4_K_M.gguf`, dated October 10; verified SHA `8480ccc295ac7379106f33ad76c978da5d9e8da0785e827786fe9d17f5a5beb5` confirms it is the old beta. Do not trust its name: the selected October 13 combined model must hash to `abfea5f878331206f65e4117157984e14803b093b634c0ed52513c0723ef0699`.
- New `tools/benchmark_fresh180.py --validate-only` passes: original frozen test SHA, 180 IDs/states/gold labels, and unchanged question schema against saved reference.
- Failure-inclusive accuracy/macro-F1 and nearest-rank p95 checks pass. Two mocked end-to-end tests pass (180 saved responses, invalid-response failure accounting, wrong-model rejection). Actual HTTP/device inference remains untested for this runner.
- Release README now distinguishes completed held-out results and corrects the mixed source licenses. Final weight redistribution terms still need review.

## Next actions

1. Transfer selected combined GGUF from the laptop path in `DESKTOP_HANDOFF_20261015.md` to `experiments/decision-followups/gguf-combined-step100/model-Q4_K_M.gguf`; verify SHA above. The old file has the same byte size: size alone is insufficient. Locate/transfer adapter, head, tokenizer and calibration archives as well before packaging a release.
2. Transfer the verified model to a distinct phone filename, e.g. `/sdcard/Download/models/reflex-1-combined-Q4_K_M.gguf`. Preserve the beta for comparisons.
3. Inspect installed decision-capable runtime, pin its binary/source hashes and save exact startup command/log. Start with the existing beta settings, including `--no-repack`, and the frozen schema. Forward localhost port 8091. Record server PID.
4. Run from repo root:

   ```sh
   python3 tools/benchmark_fresh180.py \
     --adb /Users/cbarkinozer/.cache/reflex-android/platform-tools/adb \
     --serial R68T800LZ0H \
     --device-model /sdcard/Download/models/reflex-1-combined-Q4_K_M.gguf \
     --runtime-revision '<verified revision and binary SHA>' \
     --launch-command '<exact server startup command>' \
     --pid '<server PID>' \
     --out experiments/decision-followups/a32-fresh180/run-01
   ```

   Output directory must be new. Runner verifies the phone model SHA before inference; failures count as incorrect. Fixed order, no guard/cooldown, first request separated from remaining warm sustained requests. Saves raw responses, drift/agreement against PyTorch and full battery/thermal/memory traces. HTTP duration includes transport. It does **not** measure server load time or sample in-flight peak PSS; measure those separately. USB-powered traces do not establish energy savings. Do not change schema or tune on this cohort.
5. Inspect full completion, failures, paired label/probability differences, per-route recall, latency and resource traces. Measure cold startup and peak memory separately; repeat matched sustained blocks where justified. No claims of superiority over Qwen without a matched fresh-180 baseline.
6. Update model card and GGUF manifest with actual measured results; prepare adapter/head/config/tokenizer/calibration and GGUF inventories, attribution/license notes and dataset card. Keep held-out data private and models local until explicit publication request. Do not push.

## Runtime inspection and artifact blocker

Installed Android runtime reports `0.6.0-dev`, build `11443`, commit `f0c41e016`, Clang 21.0.0 / Android aarch64. Installed file SHA-256s:

| File in `/data/local/tmp/reflex-runtime/` | SHA-256 |
| --- | --- |
| `llama-server` | `db82377125f69e87c06b402bba1af072e2a4c4cc9945946e4d6378cc213a84d7` |
| `libllama-server-impl.so` | `067ddf037540476eff509d9b6f75169f849a9459c4cab0546c6cc61a7d304730` |
| `libllama.so` | `0a1a6cdaa5889f613244e03a87fb4232a0ce9dae49b023de0759a28976026d82` |
| `libggml-cpu-android_armv8.0_1.so` | `c55abebfdb8084c242c00a87d8b2f15f96766b2bab54b57588f6b6939e563221` |

Searched this Mac's Documents, Downloads and mounted Volumes for Q4_K_M files and October 13 training archives; only the original beta model was found. The previous evaluation pod SSH endpoint `ssh6.vast.ai:10788` refuses connections. No pod was created or changed. Transfer from laptop is required before selected-model inference can begin. The phone runtime was inspected without loading the beta or starting an inference server.

## Blocker resolved via private Hugging Face repository

User supplied `cbarkinozer/Reflex-1` as the transfer source. Authenticated read succeeded at pinned revision `157a0d6ed5a218352bcd5b6bed57dbef090b151d`; privacy remained true. Main GGUF and projector were downloaded and SHA-256 verified against the existing inventory, then placed in `gguf-combined-step100/`. Main GGUF was transferred to the distinct combined filename on A32 and the device SHA verified. All nine adapter/head/tokenizer/config files were separately downloaded into ignored `data/private-hf-reflex1/` and verified byte-for-byte against `releases/reflex-1/huggingface-upload.json`. No HF or Git upload occurred.

Local model-card conflict from the laptop update was resolved retaining both corrected licenses and private-upload status; merge commit `f5767f0`.

A32 run started with the recorded historical runtime/settings; see `a32-fresh180/startup-01/launch.json`. Initial premature launch during ADB transfer failed and was preserved separately; verified complete-file restart reached ready in 2.08 s with filesystem caches retained. This is process launch/readiness, not cache-flushed cold loading. Full fixed-order benchmark output is under `a32-fresh180/run-01/`; treat results as incomplete until summary and manifest both declare completion. No Qwen comparison or app integration is implied.

## Completed selected-model A32 fresh180 run

Local result commit `c54da8e`: 164/180 correct (91.11%), macro-F1 91.06%, zero request errors. Warm mean31.38s, p50/p9532.34/33.87s; first request27.73s. Battery28.9–37.5C, AP max52.6C, max Android thermal status3. Sampled PSS max1090.65MiB, RSS high-water1093.00MiB. Per-case reference agreement94.44% (10 changes), mean per-case max absolute probability drift0.0654, maximum0.8273. Accuracy equality does not establish probability parity. Native held-out phone ECE3.10% (10 bins); Brier sum0.1338, hard-label NLL0.3178. No test-based calibration or tuning.

`a32-fresh180/run-01/inventory.json` hashes complete outputs; `parity-and-errors.json` and `errors.json` preserve all changes/errors. Main queue remains active for the matched Qwen/beta phone jobs, historical90 jobs, then full general cohorts for both decision models. Source/protocol are committed; the Mac is kept awake with caffeinate. No push, HF upload, visibility change, new GPU rental or W&B evaluation run.

## Completed schema-format control and baseline additions

Qwen raw schema-JSON (uncached) fresh180: 42/180=23.33%, zero request errors; warm mean24.94s, p50/p9525.44/26.70s. This profile is faster than the selected decision model but less accurate. It uses a different prompt/output schema than the app's original55.6% historical treatment and is explicitly a format control. No overall speed win is inferred from it.

Predeclared additions preserve the existing app prompt/grammar as a separate baseline and test prompt caching within both profiles. See CACHED_BASELINE_PLAN.md and EXISTING_ROUTER_BASELINE_PLAN.md for chronology, known scope limitations and exact unchanged-profile provenance. At the job boundary the old queue worker was stopped during beta cooldown before any beta request, then restarted with the committed14-job queue; existing complete jobs are skipped without changing their artifacts. Existing-app uncached/cached comparisons are prioritized before schema-cached and remaining beta/historical/general jobs. No new checkpoint selection or tuning on test labels.

## Completed existing-app uncached fresh180 baseline

Qwen unchanged app router:100/180=55.56%, macro-F1 53.71%, zero request errors; all-request mean41.14s, warm mean41.17s. Reflex-1:164/180=91.11%, all-request mean31.36s. Difference35.56pp; paired bootstrap95%CI27.78–43.33pp. Cached profiles and general suites remain pending. Single sustained run and prompt/policy differences limit attribution.

Queue encountered an already-exited PID between pidof and kill after this completed job. Preserved traceback in queue-console.log; fixed shutdown to tolerate confirmed process absence only, resumed completed jobs without rerunning requests. Cached-app job starts at29.4C/status0 after elapsed idle time, recorded in startup.json. New worker uses the infrastructure fix only; no prompt/model/scoring changes.

## Scope narrowed after user ETA objection

User objected to waiting3–5days. Expanded14-job queue was excessive for the immediate goal. Finish current cached existing-app fresh180 job, then stop at completed job boundary; no partial inference is interrupted. Required set is4jobs (selected native, unchanged app uncached/cached, already-completed schema format control). Ten expanded beta/historical/general Android jobs are explicitly deferred. Existing GPU general benchmark results retain their original hardware labels. `scope-update.json` preserves this decision; boundary supervisor waits for DONE cached job before terminating host workerPID17020 during next cooldown. `scope-boundary.log` and `scope-boundary-complete.json` record execution. Reports now distinguish revised `complete` from `original_expanded_queue_complete`; no deferred run is scored.

## Revised matched scope complete —11:43 Istanbul

Cached unchanged app Qwen:98/180=54.44%, zero errors, mean35.395s, warm p50/p9536.315/38.826s. Reflex-1:164/180=91.11%, mean31.364s;36.67pp accuracy gain and11.39% lower mean latency for this treatment. Cached baseline reused tokens on179/180 responses (reported input fraction22.65%). Paired accuracy difference bootstrap95%CI28.89–44.44pp. Qwen cached sampled PSS854.83MiB versus reflex1090.65MiB, and maximum battery37.5C in both; no memory/thermal/energy benefit established.

Boundary supervisor stopped expanded worker after DONE; adb confirms no llama-server remains. Final shutdown log matched existing-log prefix and is preserved. All four required jobs have180 outputs; complete inventories refreshed. No extra Android suite began. `complete:true` describes revised4-job scope only; original expanded queue remains incomplete with10 deferred jobs. Model card and dataset guide are ready locally; actual train-only dataset export still requires exact executed781-row input from Windows backup. No push, HF upload or visibility changes.

## Model card, comparison landscape and efficiency figure update

User requested the full reference comparison table retained, with model size/resource context. Restored all six published Decision Index columns (Clef, Clef-flash, Jev, DiffusionGemma Jev, Kev9B, Laya) and added Reflex-1 on its own explicitly labeled accuracy/cohort rows. Citation revisions/metadata are in releases/reflex-1/reference-benchmarks.json. Added nominal/backbone versus exact head parameter distinctions, known resource context and unknowns; no score-per-billion or cross-protocol Pareto claim. Generated PNG/SVG A32 accuracy–latency and accuracy–sampled-RAM frontier plots using tools/plot_a32_efficiency.py and saved machine-readable inputs/frontier membership.

At the user's explicit model-card update request, updated existing private HF README and seven aggregate/example/figure files in one commit, preserving private visibility. No weights, held-out queries, annotations or credentials were uploaded. Receipt releases/reflex-1/hf-model-card-update-20261015.json records the prior revision, new revision and file SHA-256 verification. No Git push or public release. Actual phone train dataset packaging still needs the exact Windows training inputs.

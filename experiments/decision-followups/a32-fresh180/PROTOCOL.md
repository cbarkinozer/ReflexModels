# A32 sequential benchmark protocol — frozen before baseline inference

Purpose: determine whether reflex-1 improves route selection over the existing Qwen raw-HTTP treatment and the original decision model on the same device. Results may support or refute that claim. No training or tuning on this evaluation.

Device: Samsung SM-A325F, serial R68T800LZ0H. Runtime Android build11443, source f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a. CPU only, 4 inference/batch threads, flash attention off, fit off, weight repacking off. Sequential requests, one model loaded at a time. USB power remains connected, so neither battery percentage nor charge-counter movement establishes energy savings.

Sequence:
1. Finish original reflex-1 fresh180 run already in progress (exact frozen schema, ctx2048).
2. Qwen3.5-0.8B Q4_K_M, same 180 IDs/order and semantic question descriptions, raw completion with JSON grammar, no chat template (ctx2048).
3. reflex-1-beta, same fresh180/schema (ctx2048).
4. reflex-1 and Qwen raw on original historical90, in sequence, with its original row schemas (ctx2048). Historical 55.6% is a reference from the earlier prompt/runtime treatment; new schema-matched raw prompts are separately documented and not mislabeled as an exact historical reproduction.
5. Full frozen typed-decisions (400 cases/2000 decisions), BANKING77 (462 cases/77 classes), CLINC150 in-domain (450 cases/150 classes), first reflex-1 then beta (ctx4096). These can exceed an 8–9 hour window; preserve full cohorts. No new sample or post-score downsampling. Existing PyTorch general results remain a separately labeled reference. Qwen phone generation baseline is not a general decision-head baseline.

For baseline raw generation the prompt includes complete state/questions/options; fixed temperature0.3, seed3407, max_tokens256, cache_prompt=false. Per-row grammar forces allowed option-key JSON and remains part of this treatment. It does not establish performance of every possible Qwen prompt, thinking configuration or tool-calling runtime. Phone route semantics include plausible future capabilities; accuracy does not demonstrate actual tool availability or final-answer quality.

Start each subsequent model job after cooldown to battery<=30C and Android thermal status0; wait at most30min for that target, then use the historical gate (<38C/status<=1) if needed and record the deviation. Each job is a sustained fixed-order sequence with no per-case cooling; record battery/AP/skin thermal data, PSS and RSS high-water mark before/after each request. File caches are retained; process startup time is not a cache-flushed cold load. Sampling between requests is not continuous peak-PSS capture.

All HTTP bodies/errors saved. Timeout300s; invalid/missing/truncated responses score as errors, never omitted. Choice predictions use API choice; noul/score discrete accuracy uses argmax probabilities, not rounded expected score. Explicit gold label takes precedence over soft gold distributions. Report per-decision and exact-record accuracy for typed; macro-F1 for phone/intents. Warm latency excludes the first request; disclose all-request latency as well. Report model probability drift and route agreement against the 180-case PyTorch reference. Any restart gets a resume record and invalidates uninterrupted sustained-timing interpretation.

Paired phone outcome differences, uncertainty and per-route results are computed after completion. Do not compare fresh180 accuracy directly to the old90-case score as a matched gain. Keep quantization/runtime drift visible. No W&B for evaluation. No uploads, Git pushes, public release, or held-out-data publication.

# Frozen fresh phone test ? 2026-10-15

All 180 human reviews are submitted and accepted: 20 per route, 90 everyday requests, 45 indirect requests, 45 boundary cases. No proposed route was changed. Review was assisted by synthetic-author suggestions; this limits independent label-validation evidence. This agent has previously seen project examples; semantic and pretraining independence are not guaranteed.

`test.jsonl` is frozen for final evaluation only. SHA-256: `2d18133e44a9f9db82fe80099b642f00cd6a12ebf643dc043d1f89f9ddfa209b`. It uses the original training route schema. `freeze-manifest.json` records provenance/hashes and zero exact overlap against 8,446 protected normalized states from archived training, validation, calibration, historical/public test data and the original review snapshot. `overlap-audit.json` contains lexical candidate screening; this does not prove semantic independence.

Human responses are preserved in `review-snapshot.json`. Candidates are in `candidates.jsonl`. The freeze script refuses to overwrite the test. No model predictions have been obtained on this test. Keep it out of training, calibration and checkpoint selection. Use a separate development dataset for subsequent improvements.

Before GPU work, commit source, annotations, frozen data, manifests and evaluation configuration. Record the exact commit in the GPU run. Large weights/backups remain outside Git with verified hashes. Export review progress using `fresh_phone_test_review.py export`; never silently refreeze after a changed annotation.

Remaining: fresh-test evaluation with per-route accuracy/macro-F1 and errors; GGUF parity; on-device timing, memory and energy assessment; conditional Hugging Face release with cards/provenance/license review.

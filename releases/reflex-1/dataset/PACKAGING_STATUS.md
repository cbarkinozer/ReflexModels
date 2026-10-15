# MOBILE_ROUTING9 release preparation

Requested repository: `cbarkinozer/MOBILE_ROUTING9` (not created or uploaded).
Related model: https://huggingface.co/cbarkinozer/Reflex-1 (private).

Prepared locally: dataset-card template, train-only exporter, provenance/isolation checks and upload/link instructions in ../DATASET_UPLOAD_GUIDE.md.

Not yet upload-ready: the exact 781 admitted training rows and executed run manifest are absent from the available Mac Documents/Downloads files. The old companion train.jsonl is not an interchangeable source. Recover the exact phone-routing-human-reviewed-v2-1080-fixedval-20261013 directory and selected run manifest from the verified Windows archive described in the guide. Source SHA must match the executed manifest before export. Author license choice remains pending.

Only the 781 training rows are intended for this package. Validation, calibration, historical tests and the separate 180-case MOBILE_ROUTING9 benchmark remain excluded. Sharing the task name does not make this training dataset the test cohort. Do not add a non-existent dataset ID to the model metadata yet.

After source recovery: run the exporter with the recorded expected SHA, copy/fill the README template, add the selected LICENSE, review the clean exported text, upload privately, verify 781 rows and nine routes, then link the actual dataset ID from the model YAML and prose. Public visibility changes remain the author's later step.

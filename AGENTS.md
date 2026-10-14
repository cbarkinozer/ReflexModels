# Repository working notes

For model experiments or rented GPU work, read and apply [EXPERIMENT_CLOSEOUT.md](EXPERIMENT_CLOSEOUT.md). Preserve configs, source/data revisions, logs, raw metrics, W&B status, provider pricing/runtime/cost records and verified local artifact backups. Update existing experiment records; never expose credentials.

W&B is for actual fine-tuning only, not smokes, diagnostics, evaluation or export. reflex-1 is available in a private Hugging Face repository at the user's request; public publication remains deferred until mobile testing succeeds.

Before starting any GPU run, export current human annotations and commit relevant source, dataset snapshots/splits, configs and manifests locally. Record the commit hash in the run. Keep credentials and large model/artifact archives outside Git with verified backup hashes. Never push or publish private held-out data as part of this checkpoint.

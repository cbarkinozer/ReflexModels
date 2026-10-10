# Experiment records and rented GPU closeout

Apply this checklist to future runs. Read the experiment's existing report, cost record and backup status before starting work; update the same records as the run progresses.

- Save model/dataset/source revisions, exact executed scripts/notebook, resolved training arguments, seed, splits and selected evaluation IDs, schema construction, calibration procedure, environment freeze and GPU/runtime details.
- Keep setup, failures, smoke/diagnostic, training, evaluation, inference and export logs. Record what was actually validated and distinguish smoke checks from full benchmark parity.
- Use W&B only for actual fine-tuning. Name project/run clearly, save run ID and URL, authentication/sync status and offline records if necessary. Do not create W&B runs for smokes, diagnostics, evaluation or export. Never store API keys in logs, reports, model cards or backups.
- Record provider, instance/host/machine IDs, rate snapshot/date, GPU and storage rates, combined base rate, network upload/download rates, runtime and cost formula. Separate estimated base cost, training-only time allocation, measured internet charges and final invoiced total. Unknown amounts remain unknown; account for rate rounding. Capture final shutdown/runtime/invoice when available. Do not infer shutdown from permission to close the pod.
- Before declaring the pod disposable, download merged weights, adapter/head, tokenizer, calibration/configs, required optimizer/scheduler/RNG checkpoint, dataset/provenance snapshots, raw metrics, full logs, executed sources and offline W&B records. Preserve export files, converter/build revision and reproducible sources if created. Exclude credentials and unnecessary environment/cache binaries.
- Verify downloaded artifacts against remote byte counts and SHA-256 inventories; save local paths, manifests and completion status. Document any excluded artifacts and how to restore. Report plainly whether remaining work requires the rented GPU.
- Keep release names and publication status explicit. reflex-1 is prepared locally; the user has deferred publishing the first version. Do not upload it merely because a card/export exists.

Reference completed run: experiments/unsloth-qwen35-08b-decision/COST.md, cost.json, REPORT.md and BACKUP.md. Its W&B run is https://wandb.ai/c-barkinozer/reflexmodels/runs/4khq3xdg.

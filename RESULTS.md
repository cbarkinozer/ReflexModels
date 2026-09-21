# Results registry

## Recorded runs

| Run | Status | Finding |
|---|---|---|
| `tokenizer_fertility_smoke_2026-09-21` | completed | Qwen3 0.6B and 1.7B share the same tokenizer. On 12 hand-authored paired smoke examples, Turkish averaged 2.59 tokens/word and English 1.18. This is a tooling sanity check, not a backbone-selection result. |
| `massive_1.0_prepare_2026-09-21` | completed | Prepared 33,042 Turkish/English records locally for fixed-label baselines. The split validator found no source-ID leakage and flagged 278 normalized-text duplicates across official splits; no records were removed. |

Each completed run writes a JSON document under `results/` containing at least:

```json
{
  "run_id": "unique-id",
  "git_commit": "sha",
  "timestamp_utc": "ISO-8601",
  "model": "identifier and revision",
  "task": "choice|binary|ordinal|tokenizer_fertility|latency",
  "language": "tr|en",
  "dataset": {"name": "...", "version": "...", "split": "..."},
  "config": "configs/...",
  "seed": 0,
  "hardware": {"cpu": "...", "ram_gb": 0},
  "metrics": {},
  "duration_seconds": 0
}
```

Use separate records for Turkish and English. Calibration records should include
NLL, Brier score, ECE, and the pre/post-temperature-scaling state.

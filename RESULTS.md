# Results registry

## Recorded runs

| Run | Status | Finding |
|---|---|---|
| `tokenizer_fertility_smoke_2026-09-21` | completed | Qwen3 0.6B and 1.7B share the same tokenizer. On 12 hand-authored paired smoke examples, Turkish averaged 2.59 tokens/word and English 1.18. This is a tooling sanity check, not a backbone-selection result. |
| `massive_1.0_prepare_2026-09-21` | completed | Prepared 33,042 Turkish/English records locally for fixed-label baselines. The split validator found no source-ID leakage and flagged 278 normalized-text duplicates across official splits; no records were removed. |
| [`berturk_tr_checkpointed`](results/berturk_tr_checkpointed/result.json) | completed | MASSIVE Turkish 60-intent fixed-label test: accuracy 0.8776, macro-F1 0.8377, calibrated NLL 0.5059, ECE 0.0201; single-request CPU latency p50 48.5 ms. Three epoch checkpoints were saved and result/prediction integrity validated. |
| [`berturk_tr_sensitivity`](results/berturk_tr_sensitivity/result.json) | completed | Sensitivity run removing 244 train and 41 dev utterances whose normalized text overlaps the unchanged MASSIVE test split: test accuracy 0.8618, macro-F1 0.8216, calibrated NLL 0.5198, ECE 0.0180; CPU latency p50 48.9 ms. This is a stricter overlap-policy comparison, not a replacement for the official-split result. |
| [`qwen3_0_6b_v0_negation`](results/qwen3_0_6b_v0_negation/result.json) | completed, preliminary | Frozen 20-item Turkish negation seed, raw V0 independent option log-probability: accuracy 0.60, pair accuracy 0.20, NLL 0.6758, p50 677.5 ms on CPU. This tiny benchmark is not a claim of broad model quality. |
| [`tokenizer_fertility_massive_local`](results/tokenizer_fertility_massive_local.json) | completed | On 11,514 MASSIVE training utterances per language: Qwen3-0.6B-Base 2.12 TR / 1.09 EN tokens per word; BERTurk 1.31 TR / 1.85 EN; XLM-R 1.51 TR / 1.19 EN. |

The first BERTurk attempt finished training but failed in post-training
calibration before saving a checkpoint or metrics. The runner was corrected,
given per-epoch disk checkpoints and an end-to-end smoke gate, then the
checkpointed run above completed. The smoke and saved-result validator both
passed. The initial failed attempt has no usable model result.

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

# Local data policy

Raw and prepared external data are local-only. Downloaded sources, extracted
archives, and generated baseline files must not be committed unless their
license and redistribution terms are explicitly reviewed.

For MASSIVE, download the official release, then run:

```powershell
python scripts/prepare_massive.py `
  --tr PATH\TO\tr-TR.jsonl `
  --en PATH\TO\en-US.jsonl `
  --output-dir data\prepared\massive
```

The adapter preserves `translated` provenance for both locales and produces
separate `tr` and `en` files for each official partition.

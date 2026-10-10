# reflex-1 experiment cost

User-provided Vast.ai dashboard snapshot on 2026-10-10: instance 55239947, host 115630, machine 95005, one RTX 4090 24 GB, 80 GB disk, on-demand. The instance was still running at the snapshot; shutdown and a final invoice are not confirmed.

| Item | Displayed rate |
| --- | --- |
| GPU | $0.334/hour |
| 80 GB disk | $0.030/hour |
| Combined base rate | $0.363/hour |
| Internet upload | $5.334/TB |
| Internet download | $2.667/TB |

At **2 hours 7 minutes**, estimated base cost is **$0.76835 (about $0.77)**: `127 / 60 × 0.363`. Internet charges are additional and cannot be calculated without measured billable usage. Displayed GPU and disk rates sum to $0.364/hour because of rounding; this estimate uses the dashboard's combined rate.

The actual 60-step training took 150.9813 seconds. Its prorated share of the displayed base rate is approximately **$0.01522**, but that is not the experiment's total cost. Setup, dependency failures, diagnostics, evaluation, model export, transfer waits and idle time also consume billed runtime. The snapshot runtime is not a final lifetime total.

`cost.json` preserves the IDs, hardware snapshot, displayed rates, calculation, unknown final billing fields and W&B link. No credentials are stored. Full artifacts and verified hashes are documented in BACKUP.md. The first model remains local; publication is deferred at the user's request.

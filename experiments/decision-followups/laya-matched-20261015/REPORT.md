# Matched Laya evaluation — 2026-10-15

Pinned English Laya 421M at 7b928d828b7b0e022f929d9bd2e44165aa270148, SDK 0.4.7, Apple M4 24GiB MPS, batch 8. Official max_len4096/head_max_len3072 overrides were registered before scoring to keep full77/150-option text. No training, calibration or checkpoint selection; no W&B, rented GPU or paid API. The SDK logged its own positive-temperature clamp for choice:11+; no manual calibration was performed, and positive temperature scaling does not change argmax choice accuracy.

| Benchmark | Cases / decisions | Base reference | Reflex-1 | Laya English |
| --- | ---: | ---: | ---: | ---: |
| MOBILE_ROUTING9 | 180 /180 | 54.44% | 91.11% | 47.22% |
| typed-decisions | 400 /2000 | 34.10% | 60.70% | 36.30% |
| BANKING77 | 462 /462 | 8.44% | 74.89% | 36.36% |
| CLINC150 | 450 /450 | 4.44% | 81.11% | 55.33% |

Every case ID, state, hard gold label and input file SHA matches the frozen cohort used for Reflex-1. Laya had zero request errors, state truncations, lost/collapsed option markers or option-text cuts under the registered profile; per-option text also fits the SDK 48-token cap. Audits of the shipping 512/192 budget show option collapse for BANKING77 and missing options/state for CLINC150, so those default budgets were not used as the comparator. Default-profile accuracy was not evaluated. No alternative profile was selected after reading scores.

Base general scores use the recorded newly initialized, unseeded FastDecisionModel head; base MOBILE_ROUTING9 uses the cached raw app router. Those paths are labeled explicitly rather than called a uniform generative zero-shot reference. Reflex-1 was adapted on disjoint task training partitions, while this Laya English checkpoint was not additionally adapted. Its separate typed-decisions specialist was not evaluated. General Reflex-1 measurements are GPU PyTorch, mobile Reflex-1 is A32 GGUF, and Laya is Mac MPS; these scores support accuracy comparisons on the frozen cohorts, not latency or energy comparisons across hardware.

Raw predictions/responses, primary/default budget audits, logs, protocol, environment and file inventories are local in this folder. Only aggregate metrics and generic benchmark definitions accompany the private Hub card; held-out input text is not uploaded. The public-safe summary is releases/reflex-1/laya-matched-summary.json.

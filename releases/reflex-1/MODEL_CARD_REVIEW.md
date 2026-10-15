# Scientific/readability review — 2026-10-15

Accepted: define Samsung Galaxy A32 immediately; explicit intended use; fix whitespace; remove deferred rows from public tables; separate calibration discussion; readable Python schema; label MOBILE_ROUTING9 (ours) and Reflex-1 (780M); retain paired confidence interval and execution/training-exposure caveats.

Corrections to the outside critique:
- Accuracy on identical cohorts can be compared across hardware, provided numerical/runtime differences and training exposure are disclosed. Latency/RAM comparisons require matched execution conditions; Laya is excluded from the A32 resource comparison.
- A single author-reviewed label policy is a legitimate task reference, but cannot establish universal intent correctness or independent annotation reliability. The card states policy alignment and reviewer exposure.
- Direct tensor counting subsequently established 779,717,956 deployed text parameters (780M rounded), including the decision head; the vision sidecar is excluded. See parameter-count.json.
- HTML markers are invisible in the rendered card and retained to preserve reproducible report generation.
- license: other is not a substitute for choosing and providing an actual license. No license grant is fabricated.

Material remaining scientific limits: freshly initialized/unseeded general base head is not prompted zero-shot Qwen; Laya and Reflex have different adaptation histories; synthetic single-reviewer test; one sequential mobile run per treatment; different routing policies; no matched mobile Laya resources or energy measurement; aggregate general metrics lack per-case Reflex outputs for paired uncertainty. More formatting cannot remove these limits.

Release blockers: author-selected model/dataset licensing, exact 781-row train export recovered from executed backup, reciprocal link after dataset exists. Do not invent a public dataset or exact training environment from the restored evaluation environment.

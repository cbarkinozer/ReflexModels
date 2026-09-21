# Decisions

| ID | Date | Decision | Rationale | Expected effect |
|---|---|---|---|---|
| D-001 | 2026-09-21 | Turkish is the primary language; English is the control language. | Turkish performance must be designed and measured from Milestone 0. | Prevents English-first evaluation drift. |
| D-002 | 2026-09-21 | Start with duplicated candidate branches. | It is the structurally correct oracle for later packed execution. | Enables exact packed-vs-duplicated correctness tests. |
| D-003 | 2026-09-21 | Select a small backbone only after fertility and baseline measurements. | Parameter count alone is not a CPU cost proxy for Turkish. | Avoids premature backbone lock-in. |
| D-004 | 2026-09-21 | Use the standalone `tokenizers` backend as a fertility-benchmark fallback. | The local Transformers package has an incompatible dependency range, but raw tokenizer encoding is sufficient because special tokens are excluded. | Unblocks tokenizer-only measurements without model weights. |
| D-005 | 2026-09-21 | Flag, rather than automatically remove, cross-split normalized-text duplicates in official datasets. | Identical short utterances can be legitimate source data; automatic removal would silently alter the official evaluation split. | Baseline reports must disclose duplicate counts and optionally run strict deduplication sensitivity checks. |
| D-006 | 2026-09-21 | Predeclare a second sensitivity split that removes train/dev text overlapping the unchanged test split. | This quantifies short-utterance overlap without changing the official score. | Every fixed-label baseline runs both `report_only` and this sensitivity policy before results are inspected. |
| D-007 | 2026-09-21 | Treat direct small causal-LM option scoring as Reflex V0, then require calibration and answerability for V1; make V2 native decision architecture conditional on measured benefit. | It separates a cheap baseline from unsupported claims of proprietary parity. | Progress is evidence-driven while retaining the Turkish CPU-first goal. |
| D-008 | 2026-09-21 | Include a final observational JEV-like comparison phase, but label any open approximation as independently designed. | Observable behavior can guide research even when proprietary mechanisms are unavailable. | Enables useful capability emulation without false provenance claims. |

Add a new row before any material deviation from `PLAN.md`, including its
evaluation impact.

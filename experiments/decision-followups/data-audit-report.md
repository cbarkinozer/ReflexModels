# Phone Routing Dataset Audit Report

**Date:** 2026-10-11  
**Dataset:** `phone_routing_v1` — 900 train / 90 calibration / 90 test  
**Auditor:** automated (Claude Code)

---

## Summary

The dataset is structurally clean. No duplicates, no cross-split contamination, balanced
classes. The key open item before the next training run is human review of gold labels
and scenario family assignments, per `phone-routing-review-sheet.csv`.

---

## 1. Structure Verification

| Split | Records | Routes | Per-route | Schema |
|---|---|---|---|---|
| train | 900 | 9 | 100 each | consistent |
| calibration | 90 | 9 | 10 each | consistent |
| test | 90 | 9 | 10 each | consistent |

**SHA-256 hashes (frozen):**
- `train.jsonl`: `76afcebcf1ae72f36514e27cfb42fbd50aaa309d3a31a411d771a5f3d8162925`
- `calibration.jsonl`: `6082bea44e25ffb9f55bfbd60a10185f1fab48424c568c2d3700b318aee5744e`
- `test.jsonl`: `ed41b456f4c7eee7b6311df19bd16f39f594016dbc9b47ffd398906ac6315dec`

**Schema fields per record:**
`id`, `split`, `state`, `route__label`, `route__confidence`, `route__probabilities`,
`label_agreement`, `factors`, `gold`, `n_questions`, `questions`, `workflow`

---

## 2. Duplicate / Leakage Audit

| Check | Result |
|---|---|
| Exact state duplicates (any split) | 0 |
| Normalized duplicates (lowercase, strip punctuation) | 0 |
| ID duplicates across splits | 0 |
| Cross-split contamination (same normalized text in ≥2 splits) | 0 |

**No leakage detected.** Test data is clean.

---

## 3. Label Quality

All 1,080 records have `total_variation=0.0` and `argmax_agree=true`. This means all
gold labels are unanimous — expected for synthetic data with deterministic generation.

`route__confidence` is 1.000 for every record (generator assigns maximum confidence).

**Implication:** The dataset cannot be filtered by label confidence since all labels
report full certainty. Human review of boundary cases (see Section 5) is the only way
to remove ambiguous examples.

---

## 4. Scenario Families

Keyword-based family assignment (heuristic — see `phone-routing-review-sheet.csv` for
full per-row annotations). **480/1080 rows (44%) are currently unclassified** by the
keyword heuristics; the actual phrases are clear to a human reader but don't match
simple keyword rules. The review sheet's `scenario_family` column is pre-filled where
possible and left as `unclassified` where not.

### Family definitions per route

**answer:** arithmetic, general_knowledge, creative, not_phone  
**get_battery_info:** battery_level, charging_status, battery_health, battery_temperature, battery_projection  
**get_device_info:** memory_ram, cpu_arch, model_manufacturer, os_version, specs  
**get_media_info:** now_playing, metadata, identify  
**get_storage_info:** available_space, usage_by_type, capacity_check  
**phone_health_check:** multi_category, overall_condition, trip_readiness, issue_diagnosis  
**save_note:** explicit_save, appointment, reference_code, credential  
**search_local_files:** by_filename, by_type, by_content, by_time  
**search_notes:** recall_specific, by_topic, list_all

### Cross-split family coverage (heuristic)

Some scenario families are represented in train but not in the test sample (10 rows/route
is too small to guarantee coverage of all families). This is expected — each route's
10 test cases cannot cover all its families.

Families found in train only (absent from test by keyword heuristic):

| Route | Train-only families |
|---|---|
| answer | creative, definition |
| get_battery_info | battery_temperature |
| get_device_info | memory_ram, cpu_arch |
| get_media_info | identify |
| get_storage_info | usage_by_type, capacity_check |
| phone_health_check | issue_diagnosis, trip_readiness |
| save_note | reference_code, appointment |
| search_local_files | by_time, by_content |

These are keyword-heuristic gaps. The actual test cases may cover these families via
different phrasing. Human review will clarify which families are genuinely absent.

---

## 5. Boundary Cases (Manual Review Needed)

These boundary pairs are highest priority for human review:

**battery vs phone_health_check:**  
`"Is my phone too warm for charging?"` → get_battery_info ✓  
`"Is my phone running hot and low on space?"` → phone_health_check ✓  
`"Is my phone healthy?"` → phone_health_check (single-word "healthy" triggers multi-category check) — review needed  
`"Tell me my battery and give me a health summary"` → phone_health_check ✓

**search_local_files vs search_notes:**  
Local files = files downloaded or saved to device storage (PDFs, photos, APKs, docs)  
Notes = text saved explicitly via save_note  
`"Find the document I downloaded"` → search_local_files ✓  
`"Find the address I saved"` → search_notes (if saved via save_note) or search_local_files (if a downloaded doc) — ambiguous

**answer vs other routes:**  
`"Can I last another two hours on this battery?"` → get_battery_info (requires live battery level) ✓  
`"How long does a phone battery usually last?"` → answer (general knowledge, no live data needed) ✓  
Cases where the question has both forms need to be checked.

---

## 6. Validation Split Plan

For the initial existing-data trial, derive validation from the 900-row train pool:
- **810 train** (90/route) + **90 validation** (10/route), stratified by route, seed=3407
- Calibration split (90 rows) used only for probability calibration after checkpoint selection
- Test split (90 rows) sealed until final candidate evaluation

Family separation across the train/val split cannot be verified until humans assign
families. After review, re-split to ensure the same families appear in both halves.

---

## 7. Open Items (Require Human Review)

From `phone-routing-review-sheet.csv`:

- [ ] Review all 480 rows with `scenario_family=unclassified` — assign family or confirm existing label
- [ ] Check boundary cases: battery_temperature (train only, 9 cases), multi_category vs overall_condition in phone_health_check
- [ ] Confirm 10 cases per route in test are representative; flag any that are trivially easy or ambiguous
- [ ] Review save_note appointment (1 case) and reference_code (4 cases) — too few for reliable generalization
- [ ] Check search_local_files `by_time` (2 cases in train only) — possibly add more before training
- [ ] After family assignment: re-split train/val to ensure family balance, then freeze

---

## 8. Recommendations for New Data Generation (Future)

Before generating paraphrases or new examples:

1. Define scenario families formally (see Section 4 definitions above)
2. Target underrepresented families: battery_temperature, battery_health, identify (media), by_time (files), appointment (notes), issue_diagnosis (health), credential (notes)
3. Generate boundary examples explicitly: battery vs health_check, search_local vs search_notes
4. Specify expected behavior for multi-tool requests before generating them
5. Split new data 80/20 by scenario family (stratified), seal 20% as new test cohort
6. Run duplicate/leakage check against existing splits before admitting new examples

---

*Review sheet: `phone-routing-review-sheet.csv` (1,080 rows, pending human review)*

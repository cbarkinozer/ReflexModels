# Completed phone-routing review audit — 2026-10-13

Snapshot SHA-256: `4ab0a79b688ec8d050f6a9b8a5da4a62ad20deb37e7fd6264ed3ef33d6b1cf2c`.
All 1,080 Argilla rows are submitted: 900 original train, 90 calibration, 90
historical test. Human responses were not edited by this audit.
The decision field is `keep` on 1,079 rows and `uncertain` on one; there are
no submitted `correct` decisions. This could mean the synthetic route labels
were largely right, but it also means this one-person review supplies little
independent evidence about route-label error rates. One `keep` row actually
changes the route and is quarantined.

## Usable route dataset

The comparison-ready export is
`pocket-agents-lab/data/prepared/phone-routing-human-reviewed-v2-1080-fixedval-20261013/`.
It keeps the original 300-review pilot's 90 validation IDs fixed, so model
comparisons use the same questions. It contains 781 reviewed training rows,
90 reviewed validation, 90 reviewed calibration, and 88 usable historical test.
The train counts are 82–89 per route (ratio 1.09); validation and calibration
have exactly 10 per route. Training/validation/calibration/test IDs and
automatic same-route template groups do not overlap. Two historical-test rows
were quarantined for unclear or incompatible scenario labels.

The strict full-review exporter cannot finalize all 1,080 rows because it
detects one `uncertain` decision, one route changed but marked `keep`, four
route/scenario mismatches, and 16 `unclear` scenarios. The comparison-ready
export excludes affected rows and records their IDs in its manifest; it does
not silently rewrite human annotations. The unreviewed v1 source is intact.

## Route and scenario checks

- **Exact duplicate conflicts:** none among all submitted requests after
  case/punctuation/spacing normalization. This does not prove semantic
  duplicates are absent.
- **Cross-route near templates:** one useful hard pair: “How much RAM is on this
  phone?” (`get_device_info`) versus “How much space is on this phone?”
  (`get_storage_info`). Their differing routes are semantically correct.
- **Scenario skew in training:** `search_notes/recall_specific` 71/88,
  `save_note/explicit_save` 67/86, `answer/general_knowledge` 67/89,
  and `phone_health_check/multi_category` 50/86. Rare scenarios include
  `get_media_info/offline_availability` 1, `search_notes/list_all` 1,
  `save_note/appointment` 2, and `phone_health_check/issue_diagnosis` 3.
  This is adequate route balance for a pilot classifier, but not adequate
  evidence for a reliable scenario classifier or rare-scenario scores.
- **`other`:** 72/1,080 submitted rows. It is a useful gap marker, but
  some repeatable clusters (general status/overview, media settings,
  file provenance) could receive dedicated scenarios if scenario prediction
  becomes a target. Keep route labels unchanged by category design alone.

Four route/scenario mismatches need human review:

| ID | Request | Submitted | Likely resolution |
|---|---|---|---|
| `pr_get_storage_info_train_016` | What type of storage is in this phone? | `get_storage_info` / `specs`, `keep` | `get_device_info` / `specs`, decision `correct` |
| `pr_get_battery_info_train_042` | Will my phone survive a two-hour drive? | `get_battery_info` / `trip_readiness` | Keep battery route and use `battery_projection`, or choose phone health if broader context intended |
| `pr_phone_health_check_train_013` | What are my phone's current vital stats? | `phone_health_check` / `general_knowledge` | `multi_category` or `overall_condition` |
| `pr_phone_health_check_test_000` | Run a complete check on my device | `phone_health_check` / `general_knowledge` | `overall_condition` or `multi_category` |

Two decision conflicts also remain: `pr_save_note_train_092` changes the route
to `search_notes` but says `keep`; `pr_phone_health_check_train_007` is
`uncertain` and genuinely lacks context. The 16 `unclear` cases are listed in
`UNCLEAR_TRIAGE.md`; 14 have a clear route and likely need a scenario change,
while two are genuinely ambiguous.

Six conservative near-template groups have differing scenario labels. The
highest-value consistency checks are:

- `pr_get_media_info_train_014`, `_011`, `_023`: BPM is `metadata`, while
  ISRC and duration were marked `now_playing`; these are all properties of
  the current track.
- `pr_get_device_info_train_029`, `_027`: “64-bit native” is `cpu_arch` while
  “64-bit” is `specs`; adopt one boundary rule.
- `pr_get_storage_info_train_005`, `_047`: both ask whether a video will fit;
  one is `capacity_check`, the other `usage_by_type`.
- `pr_get_device_info_train_087`, `_043`, `_074`: CPU is `specs`, while SoC
  and GPU were marked `cpu_arch`; check whether these ask for a chip model or
  its architecture.
- `pr_search_notes_train_011`, `_048`: project ideas are `by_topic`, ideas
  are `recall_specific`; this is a fuzzy boundary rather than a clear error.
- `pr_get_media_info_train_035`, `_008`: audio output versus active audio
  are different intents; their scenario difference is defensible.

Categories still overlap conceptually: `explicit_save` versus
`reference_code`/`credential`, `specs` versus `model_manufacturer`/`os_version`,
`now_playing` versus `metadata`, and file search by type/content/time. For
route training these are audit metadata, not model targets. If scenario
prediction is added later, define precedence or allow optional facet tags.

## CPU baseline and limits

The final fixed-validation character TF-IDF logistic baseline scores 85/90
(94.4% accuracy, 0.944 macro-F1). The same model family trained on the
300-review pilot scored 72/90 (80.0%, 0.806 macro-F1) on these **same**
validation IDs. Fourteen old errors were fixed, one new error appeared, and
four remained wrong. This supports the value of more varied reviewed training
examples; it does not validate the fine-tuned Reflex-1 model.

The historical test has already been exposed during data development, and
validation was selected from reviewed training availability. A freshly
collected, independently phrased test cohort is still required before making
a strong generalization claim. Calibration fits temperature on its own 90
cases; its in-sample ECE improvement is not an external calibration test.

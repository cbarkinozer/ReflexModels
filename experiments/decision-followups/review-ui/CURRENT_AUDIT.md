# Phone routing annotation audit — 2026-10-13

Snapshot: `review-snapshot.json` (SHA-256 `3f394e77eb19e47e1153cb41bb43d8d0ee6f5c310912f16dd4fcca620ca413c1`).
This is a point-in-time audit; export Argilla again before any later rebuild.

## Coverage and skew

- 481/1,080 submitted: 301/900 train, 90/90 calibration, 90/90 historical test.
- The 599 remaining train rows are unannotated, retaining their original synthetic
  route labels. Each source route has 120 rows. Submitted coverage by original
  route ranges from 50 (`get_media_info`) to 59 (`get_battery_info`), including evaluation.
- 480 responses say `keep`; one says `uncertain`. Two submitted routes differ from
  their source routes, including the uncertain one. This is high agreement with
  synthetic routes, but it does not establish correctness of the unreviewed rows.
- Reviewed scenario categories are much less balanced: `search_notes/recall_specific`
  44/54, `save_note/explicit_save` 43/50, `answer/general_knowledge` 42/56,
  and `search_local_files/by_type` 23/51. `get_media_info/identify` has two
  reviewed example. Scenario coverage is uneven even though routes are balanced.
- 256 reviewed scenario values differ from their original CSV categories;
  many source values were `unclassified`. Route agreement must not be taken as
  agreement on scenarios.

## Conflicts to revisit

- `pr_save_note_train_092`: “Remember the SSH key passphrase is saved in the
  manager” was changed from `save_note` to `search_notes`, but decision is `keep`.
  It needs a consistent decision and route.
- `pr_phone_health_check_train_007`: “Should I run the model now or wait?” is
  marked `uncertain`, with route `answer` and scenario `unclear`. Keep it out of
  training until its meaning is clear or exclude it.
- Route/scenario incompatibility: `pr_phone_health_check_train_013` and
  `pr_phone_health_check_test_000` use `phone_health_check` with scenario
  `general_knowledge`. Their route is plausible; the scenario should be checked.
- `pr_get_battery_info_train_042` (“Will my phone survive a two-hour drive?”)
  uses `get_battery_info` with `trip_readiness`, which belongs to
  `phone_health_check`. If this is a battery-only request, `battery_projection`
  is the closer scenario.
- Six other submitted training rows and one historical test row are tagged
  `unclear`. See the provisional export manifest for quarantined IDs.
- No normalized exact duplicate among submitted requests has conflicting route
  or scenario labels. The current conservative near-template groups also found
  no reviewed group with differing labels. This does not rule out semantic
  duplicates that the grouping method misses.

## Scenario boundaries that overlap

- `save_note/reference_code` versus `credential`: both save a code. Use
  `credential` for access secrets, `reference_code` for lookup identifiers.
  `explicit_save` overlaps both because it describes the action, not the content.
  Consider merging `reference_code` into `explicit_save` if scenarios are used
  to measure intent rather than saved content.
- `get_device_info/specs` versus `model_manufacturer` and `os_version`:
  hardware model, system-on-chip, release year, build fingerprint, and IP rating
  can be called specifications. The current categories need a written priority
  rule if scenario prediction is trained.
- `get_media_info/now_playing` versus `metadata`: “Current track info” asks for
  both the active track and its details. A stable rule is `now_playing` for
  identifying active media, `metadata` for a requested property such as duration
  or artist.
- `search_local_files/by_type`, `by_content`, and `by_time` can co-occur in one
  request. For example, `pr_search_local_files_train_092` asks for documents
  worked on this month. The current single scenario field needs a primary-cue
  rule or optional facet tags before scenario training.
- `get_storage_info/capacity_check` versus overview `other`:
  `pr_get_storage_info_train_011` (“Tell me about my storage”) was annotated
  `capacity_check`, although no proposed operation needs a fit/capacity decision.
  This is a likely scenario correction.
- `pr_search_local_files_train_099` (“Is the registration form I downloaded
  still here?”) was marked `by_content`; `by_type` is also plausible. This is a
  boundary case, not a route conflict.

These are scenario quality issues. The current route model trains on route labels,
so scenario overlap does not directly change its target. If scenario labels later
become training targets, resolve boundaries or support multiple facet tags first.

## Final reviewed-only split and bias check

The reviewed-only export is in
`pocket-agents-lab/data/prepared/phone-routing-human-only-v2-20261013-r2/`.
It contains 200 training, 90 validation, 90 calibration, and 88 usable historical
test rows. Every exported row is human reviewed. Training routes have 20–26 rows
each (largest/smallest ratio 1.30); validation and calibration have exactly 10
per route. There are no normalized exact duplicate requests and no automatic
near-template overlaps between splits or different routes in this export.

Scenario distribution remains uneven. In training, `save_note/explicit_save` is
18/20, `search_notes/recall_specific` is 20/23, and
`answer/general_knowledge` is 20/24. The validation split now includes
`reference_code`, `appointment`, `by_topic`, `arithmetic`, and `creative`, but
usually only one or two examples each. These slices cannot support reliable
per-scenario accuracy claims. Other sparse training scenarios include
`save_note/appointment` (1), `save_note/reference_code` (1),
`get_media_info/identify` (1), and `search_local_files/by_filename` (1).

The dataset is route-balanced enough for an experimental route classifier.
It is not proven unbiased: selection followed Argilla review order, the source
is synthetic, and there is no independently collected sealed test. Retain all
200 reviewed training rows rather than discarding scarce examples merely to
make every route exactly 20. Evaluate macro-F1 and per-route errors alongside
accuracy, and use a new real-phrase test set for any generalization claim.

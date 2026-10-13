# Follow-up: reduce `other` after phone review

Checkpoint on 2026-10-12, while annotation is ongoing: 54 training records had
submitted human responses. Three had human scenario category `other`:

| Record | Request | Proposed treatment |
| --- | --- | --- |
| `pr_get_storage_info_train_056` | Show me my storage status | Add `storage_overview` scenario. |
| `pr_get_storage_info_train_026` | What are my storage stats? | Add `storage_overview` scenario. |
| `pr_get_storage_info_train_031` | Is the SD card mounted? | Keep under `get_storage_info`; consider `external_storage_status` as a later scenario category. |

`storage_overview` means a general storage status request without a specific
free-space, type-breakdown, or fit/capacity question. The scenario taxonomy
describes user intent, not the capabilities of the current Android app.

The keyword suggestion function proposes `other` for 390 of 1,080 source rows:

| Proposed route | Automated `other` count |
| --- | ---: |
| answer | 31 |
| get_battery_info | 56 |
| get_device_info | 59 |
| get_media_info | 61 |
| get_storage_info | 61 |
| phone_health_check | 29 |
| save_note | 1 |
| search_local_files | 46 |
| search_notes | 46 |

These are heuristic misses, **not** 390 human `other` decisions. Many fit an
existing category when read in context. After human review, export a portable
snapshot, count submitted `other` by corrected route, inspect the actual request
and review decision, and group clear recurring intents. Keep `unclear` for
ambiguous wording. Do not exclude solely because a tool has not been implemented
yet. Add a new
category only when repeated cases share a useful meaning that existing options
do not cover. Preserve original human responses and record any recategorization
as a separate adjudication layer or a new review dataset version.

Argilla does not support changing the options of an existing published question.
Do not replace the live review UI while annotation is in progress. If additional
dropdown choices are needed, finish/export current work and create a versioned
dataset with responses restored and verified.

## Pending suggestion pass (2026-10-12)

After the initial checkpoint, route-specific rules replaced 269 pending `other`
suggestions with existing categories in the live Argilla datasets. At the end
of that pass, 83 pending `other` suggestions remained across train, calibration,
and historical test. No submitted human responses were changed. The remaining
groups include storage overview, external-storage status, file provenance, and
media playback settings (for example, volume level). The last group could become
`playback_settings` in a future dataset version. The current record "Is the
volume maxed out?" is clear `get_media_info` intent with category `other` and
should be kept for routing training.

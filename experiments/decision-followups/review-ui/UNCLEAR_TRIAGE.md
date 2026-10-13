# Unclear-label triage — 2026-10-13

Latest export: 900 train, 90 calibration, 90 historical-test submitted.
Sixteen submitted records use scenario `unclear`. These are **suggestions**;
human Argilla responses have not been changed.

Use `unclear` when the intended route itself cannot be inferred. A route can be
clear even when the request lacks an argument needed to execute the tool. Keep
that route, choose its best scenario (`other` for a broad/missing scenario),
and ask a follow-up at execution time. A future `needs_clarification` flag could
represent this separately from routing and scenario labels.

| ID suffix | Request | Suggested route / scenario | Reason |
|---|---|---|---|
| `save_note_train_068` | Note this code I need later | `save_note` / `explicit_save` | Save intent clear; code value missing. |
| `save_note_train_059` | Note this down for me | `save_note` / `explicit_save` | Save intent clear; content missing. |
| `save_note_train_053` | Save this helpline number for me | `save_note` / `explicit_save` | Save intent clear; number not supplied. |
| `get_battery_info_train_030` | Show me my battery | `get_battery_info` / `other` | Broad battery overview. |
| `get_battery_info_train_019` | I need to know my battery | `get_battery_info` / `other` | Broad battery request. |
| `get_battery_info_train_044` | Battery report please | `get_battery_info` / `other` | Broad battery overview. |
| `get_battery_info_train_012` | Battery status please | `get_battery_info` / `other` | Broad battery status. |
| `get_battery_info_train_059` | Tell me the battery status | `get_battery_info` / `other` | Broad battery status. |
| `get_battery_info_train_073` | How is my battery doing? | `get_battery_info` / `other` | Broad condition/status; `battery_health` is also plausible if policy treats “doing” as long-term condition. |
| `search_local_files_train_008` | Can you find that file for me? | `search_local_files` / `other` | File-search intent clear; identifying detail missing. |
| `search_local_files_train_038` | Find something in my downloads | `search_local_files` / `other` | Search location specified, target vague. |
| `search_local_files_train_082` | Find that file I downloaded | `search_local_files` / `other` | Search by origin; specific file missing. |
| `search_local_files_test_009` | I need to find a document on my device | `search_local_files` / `by_type` | “Document” is the only stated search cue. |
| `get_storage_info_train_066` | How much cache is accumulated on my phone? | `get_storage_info` / `usage_by_type` | Cache is a storage usage type. |
| `phone_health_check_train_007` | Should I run the model now or wait? | keep `uncertain` / `unclear` | No stable phone-data task without knowing which model or why to wait. |
| `phone_health_check_train_044` | Do a quick check before I present to the client | consider `uncertain` / `unclear` | Could mean a phone check or review of presentation material. |

The last two should stay out of training unless clarified or rewritten. The
other fourteen currently get quarantined by the reviewed-only builder because
it treats `unclear` as an unresolved review. After correcting them in Argilla,
export a new snapshot and rebuild the dataset; do not edit the snapshot JSON
by hand.

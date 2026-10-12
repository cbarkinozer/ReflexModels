# Phone routing review in Argilla

Imported the unchanged 1,080-row CSV into workspace `reflexmodels` on the existing
localhost:7900 server. Existing bender review datasets were preserved. Import
verified all record IDs, fields and counts against the CSV.

- [Training — 900 rows](http://localhost:7900/dataset/1d27fbfc-4c2c-4dcd-a796-59071967a210/annotation-mode)
- [Calibration — 90 rows](http://localhost:7900/dataset/c1760daf-5a33-43a0-a113-8d8e50a5f53e/annotation-mode)
- [Historical test — 90 rows](http://localhost:7900/dataset/6a063185-6b9a-45c2-a584-8a6b5314e8a8/annotation-mode)

Start in Training. Filter `priority` to `unclassified` or `near_duplicate`; use
`unclassified_near_duplicate` for records flagged for both. There are 480
unclassified records across the three splits, and the audit's near-duplicate
phrases are flagged on both sides of each pair.

For each request, select **Correct tool / route**, select **keep / correct /
rewrite / exclude / uncertain**, and optionally check the prepared **scenario category dropdown**. Examples:
`battery_temperature`, `available_space`, `memory_ram`, `recall_specific`.
If the existing suggestion is right, confirm it and choose keep. Click **Submit**
to save your annotation; suggestions alone do not count as reviewed labels.

Optional notes can explain a correction or rewrite. Paraphrase/template group
assignment is a separate data-preparation task, not a mandatory reviewer field. Broad scenario categories
can span splits; variants of one underlying example must share a group for later
split construction. Mark unclear cases uncertain rather than forcing a route.

The guidelines inside each dataset define the nine routes and the battery versus
overall-health boundary. The CSV's synthetic agreement/confidence is not human
validation. Annotations are a separate layer: this import does not modify the
CSV, calibration, training or historical benchmark files. The historical test is
already exposed; these annotations do not turn it into a fresh sealed test.

## Re-import and export

Use the isolated environment at `~/.cache/reflex-data-prep-venv` with argilla 2.8.0.
Credentials come from the existing private review `.env`; do not copy that file
or its API key into experiment reports or Git.

```sh
python experiments/decision-followups/argilla_phone_review.py \
  --credential-env /path/to/private/review/.env

python experiments/decision-followups/argilla_phone_review.py \
  --mode export --output-dir data/prepared/argilla-phone-review \
  --credential-env /path/to/private/review/.env
```

Re-import verifies existing records and only adds missing ones, preserving all
responses. Export saves raw annotations locally without applying corrections.
After review, inspect exported decisions and create a new reviewed dataset
version; do not silently overwrite the frozen source data.

Server-side reviews currently live in the existing Docker volumes. Export after
review and before deleting/recreating those volumes. Links and CSV hash are
recorded in `argilla-review-v3/links.json`.

## Current review UI (v3)

Scenario category remains an optional prepared dropdown; only route and review
choice are required. `offline_availability` is included. `other` means a clear
request whose category is missing; `unclear` means the request is ambiguous.
The obsolete combined `other_or_unclear` option is removed. A heuristic miss
suggests `other`; it must not automatically label clear text ambiguous.

Offline media examples are flagged unsupported: MediaSessionAccess.kt reports
package, title, artist, album and playback state, not downloads/cache/offline
availability. Review those examples for exclusion or a supported rewrite.
Category membership alone does not establish tool capability.

Argilla cannot change the number of options on an existing published question.
Replacement datasets were verified against all 1,080 source records. The six
previous v1/v2 review datasets were backed up (settings, records and responses,
SHA-256 verified), then removed from Argilla at the user's request. Only three
current phone review datasets remain. Existing bender datasets were untouched.
No saved responses were present in the retired phone datasets at cleanup.

Backup inventory: `data/prepared/argilla-cleanup-20261011T210456Z/manifest.json`.
Cleanup results: `argilla-review-v3/cleanup.json`. Original CSV and dataset splits
remain unchanged. Older links are retired; use the current links above.

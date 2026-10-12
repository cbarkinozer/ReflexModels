# Restore the phone review on another computer

The committed snapshot contains all 1,080 records and 42 saved training reviews,
including reviewer usernames, values and response status. It contains no API
keys or passwords. The old Mac's HTTP links are not portable.

## Start a fresh laptop server

From the ReflexModels repository root, with Docker Desktop running:

```sh
cp experiments/decision-followups/review-ui/.env.example \
   experiments/decision-followups/review-ui/.env
docker compose --env-file experiments/decision-followups/review-ui/.env \
  -f experiments/decision-followups/review-ui/docker-compose.yml up -d
python3 -m venv .venv
.venv/bin/pip install -r experiments/decision-followups/review-ui/requirements.txt
```

Wait for http://localhost:7900 to show the login page. The local example login
is `argilla` / `argilla123`. The compose project has separate named volumes and
binds only localhost. If another Argilla already uses port 7900, use that server
with its actual private API-key env file, or change the new server's host port
and pass the matching `--url` to both scripts. Do not remove unrelated volumes.

Import and restore:

```sh
.venv/bin/python experiments/decision-followups/argilla_phone_review.py \
  --credential-env experiments/decision-followups/review-ui/.env
.venv/bin/python experiments/decision-followups/portable_argilla_review.py restore \
  --credential-env experiments/decision-followups/review-ui/.env
```

The import prints three new local links and records them in
`argilla-review-v3/links.json`. Give the **train** link to the user. Restore maps
reviewers by username to the new server's IDs and verifies restored annotations.
Repeated restoration skips identical reviews and refuses to overwrite conflicting
local decisions. There is no need for the original Mac, llm-bender-lab credentials,
model weights or a GPU to perform this review.

## Review and save progress

Read the query; confirm/correct the route; select keep/correct/rewrite/exclude/
uncertain; click **Submit**. Categories are optional dropdowns. Start with pending
records; submitted rows are already reviewed. Guidelines explain tool boundaries.

Export after a review session and before moving computers:

```sh
.venv/bin/python experiments/decision-followups/portable_argilla_review.py export \
  --credential-env experiments/decision-followups/review-ui/.env
git add experiments/decision-followups/review-ui/review-snapshot.json
git commit -m "Save phone-routing human review progress"
git push
```

Keep `.env` private/ignored. Do not use `docker compose down -v` before exporting;
annotations live in the Docker volumes until exported. Work on one review server
at a time to avoid conflicting decisions. Snapshot export does not apply labels
to frozen training/test files or start training.

## Verification performed before handoff

- Docker Compose configuration validates; images match the running Mac setup.
- All 42 saved reviews verified against the live server without changing them.
- A fresh temporary dataset successfully restored a saved human response.
- Repeated restore skipped it and verified identical values/status.
- The temporary dataset was deleted; actual review datasets were preserved.

The full Docker stack has not been started on the laptop; startup/import there
remain the laptop agent's task.

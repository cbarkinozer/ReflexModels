"""Import phone review CSV or export annotations without changing source data.

Requires argilla==2.8.0. Credentials are read from a private env file, never saved.
Existing datasets are verified and preserved; annotations are never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUTES = ["answer", "get_battery_info", "get_device_info", "get_media_info",
          "get_storage_info", "phone_health_check", "save_note",
          "search_local_files", "search_notes"]
SCENARIOS = {
    "answer": ["arithmetic", "general_knowledge", "creative", "not_phone"],
    "get_battery_info": ["battery_level", "charging_status", "battery_health", "battery_temperature", "battery_projection"],
    "get_storage_info": ["available_space", "usage_by_type", "capacity_check"],
    "get_device_info": ["memory_ram", "cpu_arch", "model_manufacturer", "os_version", "specs"],
    "get_media_info": ["now_playing", "metadata", "identify", "offline_availability"],
    "phone_health_check": ["multi_category", "overall_condition", "trip_readiness", "issue_diagnosis"],
    "save_note": ["explicit_save", "appointment", "reference_code", "credential"],
    "search_local_files": ["by_filename", "by_type", "by_content", "by_time"],
    "search_notes": ["recall_specific", "by_topic", "list_all"],
}
SCENARIO_LABELS = sorted({label for labels in SCENARIOS.values() for label in labels}) + ["other", "unclear"]


def proposed_scenario(row):
    if row["route"] == "get_media_info" and "offline" in row["state"].casefold():
        return "offline_availability"
    if row["scenario_family"] in SCENARIO_LABELS:
        return row["scenario_family"]
    text, route = row["state"].casefold(), row["route"]
    rules = {
        "get_storage_info": [("available_space", ["free", "left", "exhaust", "almost full", "remaining", "out of space"]),
                             ("usage_by_type", ["photos", "videos", "apps", "taking up", "used"]),
                             ("capacity_check", ["capacity", "total", "fit", "enough", "install"])],
        "get_battery_info": [("battery_temperature", ["temperature", "hot", "warm", "heat"]),
                             ("charging_status", ["charging", "charger", "plug"]),
                             ("battery_projection", ["last", "hours", "until", "drain"]),
                             ("battery_health", ["health", "holding up", "condition", "degrad"]),
                             ("battery_level", ["level", "percent", "left", "remaining", "how much"])],
        "get_device_info": [("memory_ram", ["ram", "memory"]), ("os_version", ["android", "version", "operating system"]),
                            ("cpu_arch", ["cpu", "processor", "architecture", "chip"]),
                            ("model_manufacturer", ["model", "manufacturer", "brand", "what phone"])],
        "get_media_info": [("metadata", ["artist", "album", "duration", "how far", "title", "position"]),
                           ("now_playing", ["playing", "listening", "watching"]), ("identify", ["identify", "recognize"])],
        "phone_health_check": [("trip_readiness", ["trip", "travel", "flight"]),
                               ("issue_diagnosis", ["slow", "problem", "wrong", "diagnos"]),
                               ("multi_category", [" and ", "everything", "all stats"]),
                               ("overall_condition", ["health", "condition", "overall", "checkup"])],
        "save_note": [("credential", ["password", "pin", "credential"]), ("reference_code", ["code", "reference", "booking"]),
                      ("appointment", ["appointment", "meeting"]), ("explicit_save", ["save", "note", "remember", "record"])],
        "search_local_files": [("by_time", ["yesterday", "last week", "recent", "today"]),
                               ("by_type", ["pdf", "photo", "video", "apk", "document"]),
                               ("by_filename", ["named", "called", "."]), ("by_content", ["containing", "about", "contains"])],
        "search_notes": [("list_all", ["all notes", "list my notes", "every note"]),
                         ("by_topic", ["about", "related to"]), ("recall_specific", ["what did", "saved", "remember", "find"])],
        "answer": [("creative", ["poem", "story", "write", "haiku"]),
                   ("arithmetic", ["calculate", "multiply", "plus", "minus"]),
                   ("general_knowledge", ["what is", "why", "explain", "who", "define"])],
    }
    for category, phrases in rules.get(route, []):
        if any(phrase in text for phrase in phrases):
            return category
    # A heuristic miss is a taxonomy gap, not evidence of an ambiguous request.
    return "other"
NEAR_DUPLICATES = {
    "What video file is playing?", "What video is playing?",
    "What is a haiku?", "What is a haiku exactly?",
    "How far into the podcast am I?", "How far am I through the podcast?",
    "What did I note about the apartment?", "What did I note about the interview?",
    "What did I note about the appointment?",
}
GUIDELINES = """# Phone tool-selection label review

Read the request and choose the correct route. The proposed label is synthetic;
generator confidence/unanimity is not independent human validation.

- answer: general knowledge, arithmetic or prose requiring no live phone data.
- get_battery_info: battery level, charging, battery health/temperature or estimates.
- get_storage_info: capacity, free space or storage usage.
- get_device_info: model, OS, RAM or hardware specifications.
- get_media_info: currently playing media and its metadata.
- phone_health_check: overall phone condition or multiple live-data categories.
- save_note: explicitly save a note.
- search_notes: retrieve previously saved notes.
- search_local_files: find files on device storage.

Battery-only requests stay battery-specific, even if they say “health”. Broad
phone condition or battery+storage requests use phone_health_check. Flag requests
whose context is insufficient or whose single-route contract is unclear.

1. Select the corrected route (or confirm the suggestion).
2. Select keep, correct, rewrite, exclude or uncertain. Use uncertain when unsure.
3. Optionally check the prepared scenario category. Choose another dropdown option
   only if it is wrong; no category typing is required. Other means the request
   is clear but its category is missing. Unclear means the request itself is
   ambiguous. Categories help coverage analysis, not tool execution.
   Offline availability is a media category, but get_media_info cannot report
   downloads/cache/offline availability. Exclude such unsupported examples or
   propose a supported rewrite; do not approve a route based only on topic.
4. Optional notes are useful for ambiguity or a proposed rewrite. Click Submit.
   Only route and review decision are required. Suggestions alone are not reviews.

Filter priority=unclassified or near_duplicate first. Do not rewrite the original
frozen benchmark files: this review is a separate annotation layer. Historical
test data is already exposed and cannot become a fresh sealed test by relabeling.
"""


def read_env(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def fields_for(row):
    flags = []
    if row["scenario_family"] == "unclassified":
        flags.append("Scenario category needs assignment.")
    if row["state"] in NEAR_DUPLICATES:
        flags.append("Flagged near-duplicate/template overlap; review group and exclusion/rewrite.")
    if proposed_scenario(row) == "offline_availability":
        flags.append("UNSUPPORTED: get_media_info exposes title, artist, album, package and playback state; it cannot check offline availability, downloads or playlist sync. Consider exclude or a supported rewrite.")
    return {"request": row["state"],
            "source_context": f"ID: {row['id']}\nOriginal split: {row['split']}\nProposed route: {row['route']}\nPrepared scenario category: {proposed_scenario(row)} (automated proposal)\nOriginal category: {row['scenario_family']}\nGenerator: {row['generator']}\n",
            "review_flags": "\n".join(flags) or "Standard review; check the route and ambiguity.",
            "original_notes": row.get("notes") or "No prior notes."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["import", "export"], default="import")
    p.add_argument("--url", default="http://127.0.0.1:7900")
    p.add_argument("--workspace", default="reflexmodels")
    p.add_argument("--version", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--credential-env", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=ROOT / "phone-routing-review-sheet.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "argilla-review-v3")
    args = p.parse_args()
    if args.mode == "import" and args.version != 3:
        raise ValueError("Older review UIs are retired; import only version 3")
    import argilla as rg
    client = rg.Argilla(api_url=args.url, api_key=read_env(args.credential_env)["ARGILLA_API_KEY"])
    workspace = client.workspaces(name=args.workspace)
    if workspace is None:
        workspace = rg.Workspace(name=args.workspace, client=client).create()
    with args.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1080 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected 1080 unique CSV rows")
    by_split = collections.defaultdict(list)
    for row in rows:
        if row["route"] not in ROUTES or row["split"] not in ("train", "calibration", "test"):
            raise ValueError("Invalid route or split")
        by_split[row["split"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    links = {}
    for split in ("train", "calibration", "test"):
        suffix = "historical-test" if split == "test" else split
        name = f"phone-routing-v1-{suffix}-review" + (f"-v{args.version}" if args.version > 1 else "")
        dataset = client.datasets(name=name, workspace=args.workspace)
        if args.mode == "export":
            if dataset is None:
                raise ValueError(f"Missing dataset {name}")
            records = dataset.records.to_list(flatten=False)
            (args.output_dir / f"{suffix}-annotations.json").write_text(json.dumps(records, indent=2, default=str) + "\n")
        else:
            if dataset is None:
                settings = rg.Settings(guidelines=GUIDELINES, fields=[
                    rg.TextField(name="request", title="User request"),
                    rg.TextField(name="source_context", title="Proposed label and provenance"),
                    rg.TextField(name="review_flags", title="Review priority"),
                    rg.TextField(name="original_notes", title="Original notes"),
                ], questions=[
                    rg.LabelQuestion(name="correct_route", title="Correct tool / route", labels=ROUTES, required=True),
                    rg.LabelQuestion(name="review_decision", title="What should happen to this example?",
                                     labels=["keep", "correct", "rewrite", "exclude", "uncertain"], required=True),
                    rg.LabelQuestion(name="scenario_family", title="Scenario category — optional; prepared for you",
                                     labels={label: label.replace("_", " ").title() for label in SCENARIO_LABELS}, required=False),
                    rg.TextQuestion(name="review_notes", title="Reason, ambiguity or proposed rewrite", required=False),
                ], metadata=[
                    rg.TermsMetadataProperty(name="route", options=ROUTES),
                    rg.TermsMetadataProperty(name="priority", options=["standard", "unclassified", "near_duplicate", "unclassified_near_duplicate"]),
                    rg.TermsMetadataProperty(name="suggested_family"),
                ])
                dataset = rg.Dataset(name=name, workspace=args.workspace, settings=settings, client=client).create()
            existing = {str(record.id): record for record in dataset.records}
            expected = {row["id"]: fields_for(row) for row in by_split[split]}
            for external_id, record in existing.items():
                if external_id not in expected or dict(record.fields) != expected[external_id]:
                    raise ValueError(f"Existing dataset differs from CSV: {name}/{external_id}")
            records = []
            for row in by_split[split]:
                if row["id"] in existing:
                    continue
                priority = []
                if row["scenario_family"] == "unclassified": priority.append("unclassified")
                if row["state"] in NEAR_DUPLICATES: priority.append("near_duplicate")
                suggestions = [rg.Suggestion(question_name="correct_route", value=row["route"], agent="synthetic-generator")]
                suggestions.append(rg.Suggestion(question_name="scenario_family", value=proposed_scenario(row), agent="keyword-heuristic-v2"))
                records.append(rg.Record(id=row["id"], fields=fields_for(row),
                    metadata={"route": row["route"], "priority": "_".join(priority) or "standard",
                              "suggested_family": proposed_scenario(row)}, suggestions=suggestions))
            if records:
                dataset.records.log(records)
            actual = list(dataset.records)
            if len(actual) != len(expected) or {str(r.id): dict(r.fields) for r in actual} != expected:
                raise ValueError(f"Imported record verification failed: {name}")
        links[split] = {"name": name, "id": str(dataset.id), "records": len(by_split[split]),
                        "url": f"{args.url}/dataset/{dataset.id}/annotation-mode"}
    manifest = {"workspace": args.workspace, "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                "datasets": links, "original_data_modified": False,
                "test_status": "Historical/exposed only; no new sealed test imported"}
    (args.output_dir / "links.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

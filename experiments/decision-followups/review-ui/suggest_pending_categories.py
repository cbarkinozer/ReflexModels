"""Refine automated `other` suggestions without changing human responses.

Run without --apply to inspect counts. Only records with no human response and
an existing `other` scenario suggestion are eligible for an update.
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from argilla_phone_review import SCENARIO_LABELS, read_env  # noqa: E402


def matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


RULES = {
    "answer": [
        ("creative", r"recommend|rhyme|proofread|rephrase|compose|draft|entertain|product description"),
        ("arithmetic", r"^convert \d|how many seconds in|how many ounces in"),
        ("general_knowledge", r"what year|how many|how do i|what should|how was|what time|can you speak|pronounce|when was|how do you say|give me a .+routine"),
    ],
    "get_battery_info": [
        ("battery_temperature", r"cool enough|\btemperature\b|\bhot\b|\bheat\b"),
        ("charging_status", r"being charged|on charge|fully charged|reached 100%|charge at maximum|amperage going into|rising or falling"),
        ("battery_health", r"capacity still full|indicator accurate|behaving normally|need replacing|performance good|getting worse|screen-on time|background activity"),
        ("battery_projection", r"can i (start|browse|make|run)|will (my phone|this phone)|survive|enough for|enough juice|for (an hour|today|the meeting|45 minutes|30-minute|a two-hour)|before i leave|too low to record"),
        ("battery_level", r"almost dead|below \d|above \d|exact charge|running low|about to die|battery status green|battery soon|charge range|battery low"),
    ],
    "get_device_info": [
        ("cpu_arch", r"arm64|armv|64-bit|\bsoc\b|\bgpu\b|modem"),
        ("os_version", r"build number|api level|security patch|custom rom|kernel|what system|device fingerprint|phone rooted"),
        ("model_manufacturer", r"kind of phone|product name|what device|samsung|manufactur|who made|release year|exact device|device identification"),
        ("specs", r"device info|tell me about this device|hardware|ip rating|sensor|5g|dlss|wifi|wi-fi|weight|dimension|hdr|charging port|fingerprint sensor|imei|nfc|bluetooth|carrier|headphone jack|sim card|flagship|mid-range|fcc id|device details|geekbench|end-of-life"),
    ],
    "get_media_info": [
        ("metadata", r"bitrate|hours into|how long|time is left|timestamp|chapter|who is|who sings|performing|featured|narrating|produced|record label|released|what year|genre|explicit|lossless|drm|total length|remix|original|spotify or|apple music|what key"),
        ("now_playing", r"song that.s on|movie is on|track is on|what book is this audiobook|what is the current|current song|what audio is active|what is being cast|media info please|media status|what is playing|name the song"),
    ],
    "get_storage_info": [
        ("usage_by_type", r"breakdown|app caches|how much cache|reserved by the system"),
        ("capacity_check", r"is there space for|is there room to|can i (store|download|sync|record)|will i be able to|do i have space for|can my phone handle more data|can i run an ai model|could.*fit|what is the internal storage size|how many gigabytes of storage|64gb or 128gb|how much storage is on my sd card|combined storage"),
        ("available_space", r"running out|critically low|low storage|low on disk|near the warning|warning threshold|disk space situation|storage space adequate|need to delete|clear app caches|exceeded my storage quota|storage holding up|storage okay|how close am i|enough space"),
    ],
    "phone_health_check": [
        ("multi_category", r"battery.*storage|storage.*battery|battery, storage|all the numbers|all the phone stats|all the key phone metrics|complete check|full report|comprehensive|performance metrics|phone vitals|current vital stats"),
        ("trip_readiness", r"long day out|hike|no charging|go\?"),
        ("overall_condition", r"heavy use|inference task|ai model|local model|ml task|long ai task|phone thoroughly|phone top to bottom|thorough phone check|phone in a good state|phone suitable|phone prepared|phone ready|run the model|device check"),
        ("issue_diagnosis", r"danger of failing|in trouble|problem"),
    ],
    "save_note": [("explicit_save", r"write this down")],
    "search_local_files": [
        ("by_time", r"from this week|earlier|haven.t opened in a long time"),
        ("by_type", r"zip archives|epub files|excel files|midi files|gif files|text files|markdown files|sqlite database|log files|voice memos|screenshot|spreadsheet|scan i took"),
        ("by_content", r"in the content|mentioning|with .+ in (them|it)|files for '|files for \"|files for the phrase|files for 'visa|files for 'insurance|design mockup|expenses|architecture diagram|copy of my diploma|presentation i exported|cv file|backup of my notes"),
        ("by_filename", r"backup file|config file|file with '|spreadsheet with|file i made|file i saved"),
    ],
    "search_notes": [
        ("list_all", r"what is in my notes|what have i noted|what information did i save"),
        ("by_topic", r"what brainstorm notes|what project ideas|what ideas did i save|what action items|what is on my grocery list|what items did i save to buy|shopping list note|what food items"),
        ("recall_specific", r"what|when|did i|check if|show me"),
    ],
}


def refined_category(route: str, request: str) -> str | None:
    for category, pattern in RULES.get(route, []):
        if matches(request, pattern):
            assert category in SCENARIO_LABELS
            return category
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-env", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:7900")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import argilla as rg

    rows = {row["id"]: row for row in csv.DictReader((ROOT / "phone-routing-review-sheet.csv").open(encoding="utf-8", newline=""))}
    client = rg.Argilla(api_url=args.url, api_key=read_env(args.credential_env)["ARGILLA_API_KEY"])
    for split in ("train", "calibration", "historical-test"):
        dataset = client.datasets(name=f"phone-routing-v1-{split}-review-v3", workspace="reflexmodels")
        if dataset is None:
            raise ValueError(f"Missing {split} dataset")
        updates = []
        counts = collections.Counter()
        for record in dataset.records:
            counts["all"] += 1
            if record.responses.api_models():
                counts["reviewed_skipped"] += 1
                continue
            scenario = next((s for s in record.suggestions if s.question_name == "scenario_family"), None)
            if scenario is None or scenario.value != "other":
                continue
            counts["pending_other"] += 1
            row = rows[str(record.id)]
            category = refined_category(row["route"], row["state"])
            if category is None:
                counts["left_other"] += 1
                continue
            counts[category] += 1
            if args.apply:
                suggestions = [rg.Suggestion(question_name=s.question_name, value=(category if s.question_name == "scenario_family" else s.value), agent=("semantic-rules-v1" if s.question_name == "scenario_family" else s.agent)) for s in record.suggestions]
                updates.append(rg.Record(id=str(record.id), fields=dict(record.fields), metadata=record.metadata.to_dict(), suggestions=suggestions))
        if updates:
            dataset.records.log(updates)
        print(split, dict(counts), "applied" if args.apply else "dry-run")


if __name__ == "__main__":
    main()

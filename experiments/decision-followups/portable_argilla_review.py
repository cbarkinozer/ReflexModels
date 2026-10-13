"""Export/restore portable review snapshots. No passwords/API keys are exported."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from argilla_phone_review import ROOT, read_env


def canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def response_models(record):
    return [response.model_dump(mode="json") for response in record.responses.api_models()]


def export_snapshot(client, workspace, csv_path, path):
    users = {str(user.id): user.username for user in client.users}
    datasets = []
    for split in ("train", "calibration", "historical-test"):
        name = f"phone-routing-v1-{split}-review-v3"
        dataset = client.datasets(name=name, workspace=workspace)
        if dataset is None:
            raise ValueError(f"Missing {name}")
        records = []
        for record in dataset.records:
            records.append({"id": str(record.id), "fields": dict(record.fields),
                "metadata": record.metadata.to_dict(),
                "suggestions": [{"question_name": s.question_name, "value": s.value, "agent": s.agent} for s in record.suggestions],
                "responses": response_models(record)})
        datasets.append({"name": name, "records": records,
                         "reviewed_records": sum(bool(r["responses"]) for r in records)})
    snapshot = {"version": 1, "exported_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "workspace": workspace, "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "users": users, "datasets": datasets}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Validate serializable responses and counts before replacing the snapshot.
    restored = json.loads(temporary.read_text(encoding="utf-8"))
    assert sum(len(d["records"]) for d in restored["datasets"]) == 1080
    temporary.replace(path)
    return snapshot


def restore_annotations(client, dataset, records, users, rg):
    local_users = {user.username: str(user.id) for user in client.users}
    existing = {str(record.id): record for record in dataset.records}
    pending, expected = [], {}
    for raw in records:
        if raw["id"] not in existing or dict(existing[raw["id"]].fields) != raw["fields"]:
            raise ValueError(f"Record missing/different: {raw['id']}")
        if not raw["responses"]:
            continue
        responses, mapped = [], []
        for saved in raw["responses"]:
            username = users[saved["user_id"]]
            if username not in local_users:
                raise ValueError(f"Create reviewer {username} locally before restore")
            user_id = local_users[username]
            mapped.append({**saved, "user_id": user_id})
            for question, value in saved["values"].items():
                responses.append(rg.Response(question_name=question, value=value["value"],
                                             user_id=user_id, status=saved["status"]))
        expected[raw["id"]] = sorted(mapped, key=lambda r: r["user_id"])
        actual = sorted(response_models(existing[raw["id"]]), key=lambda r: r["user_id"])
        if actual:
            if canonical(actual) != canonical(expected[raw["id"]]):
                raise ValueError(f"Conflicting local review preserved: {raw['id']}")
            continue
        pending.append(rg.Record(id=raw["id"], fields=raw["fields"], metadata=raw["metadata"],
                                 responses=responses))
    if pending:
        dataset.records.log(pending)
    verified = {str(r.id): sorted(response_models(r), key=lambda response: response["user_id"])
                for r in dataset.records if str(r.id) in expected}
    if canonical(verified) != canonical(expected):
        raise ValueError("Restored response verification failed")
    return len(pending), len(expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["export", "restore"])
    p.add_argument("--credential-env", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:7900")
    p.add_argument("--workspace", default="reflexmodels")
    p.add_argument("--snapshot", type=Path, default=ROOT / "review-ui/review-snapshot.json")
    args = p.parse_args()
    import argilla as rg
    client = rg.Argilla(api_url=args.url, api_key=read_env(args.credential_env)["ARGILLA_API_KEY"])
    csv_path = ROOT / "phone-routing-review-sheet.csv"
    if args.mode == "export":
        snapshot = export_snapshot(client, args.workspace, csv_path, args.snapshot)
        print(json.dumps({"snapshot": str(args.snapshot), "reviewed": {
            d["name"]: d["reviewed_records"] for d in snapshot["datasets"]}}, indent=2))
        return
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != snapshot["csv_sha256"]:
        raise ValueError("Snapshot CSV differs from checkout")
    for saved in snapshot["datasets"]:
        dataset = client.datasets(name=saved["name"], workspace=args.workspace)
        if dataset is None:
            raise ValueError("Run argilla_phone_review.py import first")
        imported, verified = restore_annotations(client, dataset, saved["records"], snapshot["users"], rg)
        print(f"{saved['name']}: restored {imported}, verified {verified} reviewed rows")


if __name__ == "__main__":
    main()

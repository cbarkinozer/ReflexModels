import json, hashlib, tarfile
from pathlib import Path
root = Path("/workspace/reflex-decision")
secrets = json.loads((root / ".credentials.json").read_text())
paths = [root / "runs/notebook-60steps", root / "wandb/offline-run-20261010_165410-4khq3xdg"]
paths += [root / name for name in ["sft.log", "sft-wandb-auth-failed.log", "setup.log", "fix.log", "environment-freeze.txt", "kernels-explicit.log", "fla.log", "einops.log", "fla-revert.log", "verify-final.log", "run_notebook.py", "verify_final.py", "Qwen3_5_08B-Decision.ipynb"]]
paths += [p for name in ["smoke", "overfit", "overfit300"] for p in (root / "runs" / name).glob("*.json")]
def included(path):
    rel = path.relative_to(root)
    return not any(part in ("merged", "checkpoints") for part in rel.parts)
files = [f for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file() and included(f)]
inventory = {}
for f in files:
    data = f.read_bytes()
    if any(value.encode() in data for value in secrets.values()):
        raise RuntimeError("Credential found in artifact: " + str(f.relative_to(root)))
    inventory[str(f.relative_to(root))] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
(root / "artifact-inventory.json").write_text(json.dumps(inventory, indent=2))
with tarfile.open(root / "artifacts-portable.tar", "w") as archive:
    for f in files: archive.add(f, arcname=str(f.relative_to(root)), recursive=False)
    archive.add(root / "artifact-inventory.json", arcname="artifact-inventory.json")
digest = hashlib.sha256((root / "artifacts-portable.tar").read_bytes()).hexdigest()
(root / "artifacts-portable.sha256").write_text(digest + "\n")
print("Credential scan passed; backed up", len(files), "files; archive SHA256", digest)

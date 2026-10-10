"""Snapshot completed evaluations, all diagnostic logs/configs and resumable checkpoint."""
import json,hashlib,tarfile
from pathlib import Path
root=Path('/workspace/reflex-decision')
assert json.loads((root/'runs/balanced-benchmarks/complete.json').read_text())['completed']
keys=[v.encode() for v in json.loads((root/'.credentials.json').read_text()).values() if v]
files=set()
for pattern in ['*.log','*.py','*.sh','*.ipynb','*freeze*.txt']:
 files.update(p for p in root.glob(pattern) if p.is_file())
for run in (root/'runs').iterdir():
 if run.is_dir():
  files.update(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.log','.csv') and 'merged' not in p.parts and 'adapter' not in p.parts)
files={p for p in files if 'checkpoints' not in p.parts}
checkpoints=[p for p in (root/'runs/notebook-60steps/checkpoints').rglob('*') if p.is_file()]
merged=[p for p in (root/'runs/notebook-60steps/merged').rglob('*') if p.is_file()]
def digest(p):
 h=hashlib.sha256();carry=b''
 with p.open('rb') as f:
  while True:
   b=f.read(4*1024*1024)
   if not b:break
   if any(k in carry+b for k in keys): raise RuntimeError('Credential found in '+str(p.relative_to(root)))
   h.update(b);carry=b[-256:]
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
inventory={str(p.relative_to(root)):digest(p) for p in sorted(files)}
(root/'supplement-inventory.json').write_text(json.dumps(inventory,indent=2))
(root/'merged-inventory.json').write_text(json.dumps({str(p.relative_to(root/'runs/notebook-60steps/merged')):digest(p) for p in merged},indent=2))
(root/'checkpoint-inventory.json').write_text(json.dumps({str(p.relative_to(root/'runs/notebook-60steps/checkpoints')):digest(p) for p in checkpoints},indent=2))
with tarfile.open(root/'artifacts-supplement.tar','w',dereference=True) as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(root)),recursive=False)
 for name in ['supplement-inventory.json','merged-inventory.json','checkpoint-inventory.json']:t.add(root/name,arcname=name)
print('Credential scan passed;',len(files),'supplement files;',len(merged),'merged model files.',flush=True)

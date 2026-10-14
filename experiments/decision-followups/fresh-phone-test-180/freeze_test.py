"""Freeze fully reviewed fresh phone test; never use for training or selection."""
import collections,datetime,hashlib,json,re,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BACKUP=ROOT.parents[3]/'pocket-agents-lab/runs/decision-followups-vast-20261013/heldout-20261014/heldout-closeout-20261014.tar.gz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def main():
 snapshot=ROOT/'review-snapshot.json'; candidates=ROOT/'candidates.jsonl';target=ROOT/'test.jsonl'
 if target.exists():raise FileExistsError('Frozen test already exists; do not overwrite')
 review=json.loads(snapshot.read_text(encoding='utf-8'))
 proposed={r['id']:r for r in map(json.loads,candidates.read_text(encoding='utf-8').splitlines())}
 old=set(); schema=None
 with tarfile.open(BACKUP) as archive:
  for member in archive:
   if member.isfile() and member.name.startswith('reflex-decision/data/') and member.name.endswith('.jsonl'):
    records=[json.loads(s) for s in archive.extractfile(member).read().decode().splitlines() if s.strip()]
    old.update(norm(r['state']) for r in records)
    if 'phone-routing-human-reviewed' in member.name and member.name.endswith('/train.jsonl'):schema=json.loads(records[0]['questions'])
 historical=json.loads((ROOT.parent/'review-ui/review-snapshot.json').read_text(encoding='utf-8'))
 old.update(norm(r['fields']['request']) for d in historical['datasets'] for r in d['records'])
 if schema is None:raise ValueError('Training route schema missing')
 labels=list(schema['route']['criteria']);rows=[];seen=set();changes=[]
 for r in review['records']:
  responses=[v for v in r['responses'] if v['status']=='submitted']
  if len(responses)!=1:raise ValueError('Require one submitted review: '+r['id'])
  v=responses[0]['values'];route=v['correct_route']['value']
  if v['review_decision']['value']!='accept' or v.get('replacement_request',{}).get('value'):raise ValueError('Resolve revision/exclusion first: '+r['id'])
  text=r['fields']['request'];n=norm(text)
  if route not in labels or n in old or n in seen:raise ValueError('Invalid route or text overlap: '+r['id'])
  seen.add(n)
  if route!=proposed[r['id']]['candidate_route']:changes.append(r['id'])
  probabilities={label:float(label==route) for label in labels}
  rows.append({'id':r['id'],'workflow':'phone_routing','split':'test','state':text,'questions':json.dumps(schema,sort_keys=True),'gold':json.dumps({'route':{'type':'choice','label':route,'probabilities':probabilities}},sort_keys=True),'n_questions':1,'route__label':route,'coverage':proposed[r['id']]['coverage'],'source':'fresh_synthetic_human_validated_20261015'})
 counts=collections.Counter(r['route__label'] for r in rows)
 if len(rows)!=180 or set(counts)!=set(labels) or any(v!=20 for v in counts.values()):raise ValueError('Route balance differs; adjudicate without forcing labels')
 content=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in rows)
 target.write_text(content,encoding='utf-8',newline='\n')
 (ROOT/'schema.json').write_text(json.dumps(schema,indent=2)+'\n',encoding='utf-8')
 manifest={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':180,'route_counts':dict(counts),'coverage_counts':dict(collections.Counter(r['coverage'] for r in rows)),'test_sha256':sha(target),'review_snapshot_sha256':sha(snapshot),'candidate_sha256':sha(candidates),'schema_sha256':sha(ROOT/'schema.json'),'exact_overlap':0,'protected_normalized_states':len(old),'label_changes':changes,'status':'frozen_test_not_model_evaluated','allowed_use':'Final evaluation only; no training, calibration or checkpoint selection','limitations':['Synthetic author previously saw project examples','Human review used author suggestions; possible anchoring','Lexical screening does not establish semantic or pretraining independence'],'training_source_archive_sha256':'883959885b9280506688ab33fdeac3560fdef13ddcf4ed37d9105ff750b98a90'}
 (ROOT/'freeze-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

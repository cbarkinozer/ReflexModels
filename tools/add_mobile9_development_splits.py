#!/usr/bin/env python3
"""Package exact development splits from the verified Git source; no network writes."""
from pathlib import Path
import collections,hashlib,itertools,json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'data/prepared/hf-mobile-routing9'
    prefix='experiments/decision-followups/phone-training-781/'
    revision=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    def blob(name):return subprocess.check_output(['git','show',revision+':'+prefix+name],cwd=ROOT)
    source_manifest=json.loads(blob('dataset-manifest.json'))
    p=out/'manifest.json';m=json.loads(p.read_text());m['development_splits_source_revision']=revision
    for split in ['validation','calibration']:
        raw=blob(split+'.jsonl');expected=source_manifest['splits'][split]
        assert hashlib.sha256(raw).hexdigest()==expected['sha256']
        rows=[json.loads(l) for l in raw.splitlines()];assert len(rows)==90
        clean=[]
        for x in rows:
            label=x['route__label'];assert json.loads(x['gold'])['route']['label']==label
            clean.append({k:x[k] for k in ['id','state','questions','gold','n_questions']}|{'route':label,'source':'synthetic-human-reviewed-phone-routing'})
        target=out/f'data/{split}.jsonl';target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in clean))
        counts=dict(collections.Counter(x['route'] for x in clean));assert set(counts.values())=={10}
        m[split]={'rows':90,'route_counts':counts,'source_sha256':expected['sha256'],'export_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    splits={s:[json.loads(l) for l in (out/f'data/{s}.jsonl').read_text().splitlines()] for s in ['train','validation','calibration','test']}
    norm=lambda s:re.sub(r'\W+','',s.casefold())
    checks=[]
    for a,b in itertools.combinations(splits,2):
        ids=len({x['id'] for x in splits[a]}&{x['id'] for x in splits[b]});texts=len({norm(x['state']) for x in splits[a]}&{norm(x['state']) for x in splits[b]});assert ids==texts==0
        checks.append({'splits':[a,b],'overlapping_ids':ids,'overlapping_normalized_requests':texts})
    m.update(published_splits=list(splits),excluded_splits=['historical_test88'],missing_exact_release_inputs=[],split_isolation=checks)
    p.write_text(json.dumps(m,indent=2)+'\n')
    p=ROOT/'releases/reflex-1/dataset/HF_DATASET_CARD.md';s=p.read_text().replace('  - split: test','  - split: validation\n    path: data/validation.jsonl\n  - split: calibration\n    path: data/calibration.jsonl\n  - split: test')
    s=s.replace('- n<1K','- 1K<n<10K').replace('**961 synthetic, human-reviewed examples: 781 training and 180 test examples**','**1,141 synthetic, human-reviewed examples: 781 training, 90 validation, 90 calibration and 180 test examples**')
    s=s.replace('| Label | Train | Test |','| Label | Train | Validation | Calibration | Test |').replace('| --- | ---: | ---: |','| --- | ---: | ---: | ---: | ---: |',1)
    s=re.sub(r'^(\| `[^`]+` \| \d+ \|) 20 \|$',r'\1 10 | 10 | 20 |',s,flags=re.M).replace('| **Total** | **781** | **180** |','| **Total** | **781** | **90** | **90** | **180** |')
    s=s.replace('Separate validation and calibration sets of 90 examples each exist in the original experiment but are **not included in this release yet**.','Validation supports development and checkpoint selection; calibration is reserved for fitting probability calibration. Both preserve the original 90-example split assignments.')
    s=s.replace('No ID or normalized-request overlap was found between the released train and test splits.','No ID or normalized-request overlap was found across any pair of released splits.')
    p.write_text(s);(out/'README.md').write_text(s)
    print(json.dumps({'source_revision':revision,'splits':{s:len(v) for s,v in splits.items()},'pairwise_isolation_checks':checks},indent=2))
if __name__=='__main__':main()

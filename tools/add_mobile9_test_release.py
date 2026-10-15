#!/usr/bin/env python3
"""Add frozen MOBILE_ROUTING9 test to a local release folder; no network writes."""
from pathlib import Path
import collections, hashlib, json, re
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'data/prepared/hf-mobile-routing9'
    source=ROOT/'experiments/decision-followups/fresh-phone-test-180/test.jsonl'
    expected='2d18133e44a9f9db82fe80099b642f00cd6a12ebf643dc043d1f89f9ddfa209b'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    train=[json.loads(l) for l in (out/'data/train.jsonl').read_text().splitlines()]
    original=[json.loads(l) for l in source.read_text().splitlines()]
    assert len(original)==180 and len({x['id'] for x in original})==180
    norm=lambda s:re.sub(r'\W+','',s.casefold())
    assert not {x['id'] for x in train}&{x['id'] for x in original}
    assert not {norm(x['state']) for x in train}&{norm(x['state']) for x in original}
    clean=[]
    for x in original:
        label=json.loads(x['gold'])['route']['label']
        clean.append({k:x[k] for k in ['id','state','questions','gold','n_questions']}|{'route':label,'source':'synthetic-human-reviewed-phone-routing'})
    counts=dict(collections.Counter(x['route'] for x in clean));assert set(counts.values())=={20}
    target=out/'data/test.jsonl';target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in clean))
    p=out/'manifest.json';m=json.loads(p.read_text());m.update(published_splits=['train','test'],excluded_splits=['validation','calibration','historical_test'],missing_exact_release_inputs=['validation: 90 rows','calibration: 90 rows'],test={'rows':180,'route_counts':counts,'source_sha256':expected,'export_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'preserved_fields':['id','state','questions','gold','n_questions'],'purpose':'Frozen cohort underlying model-card MOBILE_ROUTING9 scores; never used for training/calibration/checkpoint selection.'});p.write_text(json.dumps(m,indent=2)+'\n')
    p=out/'README.md';s=p.read_text().replace('    path: data/train.jsonl','    path: data/train.jsonl\n  - split: test\n    path: data/test.jsonl')
    s=s.replace('**781 training examples across nine phone-assistant routes.**','**781 training examples and 180 frozen test examples across nine phone-assistant routes.**')
    s=s.replace('this train-only package does **not** contain the separate 180-case evaluation cohort scored in the model card.','the `test` split contains the exact 180-case evaluation cohort scored in the model card.')
    s=s.replace('and does not contain the benchmark questions.','and includes the phone benchmark as a separate `test` split.')
    a=s.index('Only training rows are included.');b=s.index('\n\n```python',a)
    s=s[:a]+'''| Split | Rows | Role |
| --- | ---: | --- |
| train | 781 | Executed phone training component |
| test | 180 | Frozen MOBILE_ROUTING9 evaluation, 20 cases per route |

Exact validation (90) and calibration (90) files are not yet available in this release; they will be added without reshuffling when recovered. Historical development-influenced test rows remain excluded. The 180-case test was reviewed and frozen after checkpoint selection. Labels reflect the author's routing policy; the reviewer saw proposed synthetic labels. Do not train, calibrate or select checkpoints on test rows to reproduce these results. Once this benchmark is used for development, use additional unseen data for future generalization claims. Zero normalized-request and ID overlap was verified between the released train and test splits; semantic/template independence is not guaranteed.'''+s[b:]
    p.write_text(s)
    print(json.dumps({'train':781,'test':180,'missing':['validation90','calibration90'],'test_route_counts':counts},indent=2))
if __name__=='__main__':main()

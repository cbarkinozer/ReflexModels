#!/usr/bin/env python3
"""Prepare a local TRAIN-ONLY HF dataset from exact training-run inputs.

No network calls. Does not infer missing training splits from review snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROUTES={'answer','get_device_info','get_battery_info','get_storage_info','get_media_info','search_local_files','search_notes','save_note','phone_health_check'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def decode(x): return json.loads(x) if isinstance(x,str) else x


def prepare(source, expected, output):
    train=source/'train.jsonl'
    if sha(train)!=expected: raise ValueError('Training file SHA differs from the executed run manifest')
    rows=[json.loads(l) for l in train.read_text().splitlines() if l.strip()]
    if len(rows)!=781: raise ValueError('Selected run used 781 phone training rows; input is a different version')
    protected=set()
    for name in ['val.jsonl','calibration.jsonl','test.jsonl']:
        path=source/name
        if not path.exists(): raise ValueError('Missing split for isolation check: '+name)
        for line in path.read_text().splitlines():
            if line.strip():protected.add(json.loads(line)['id'])
    clean=[];ids=set();counts={r:0 for r in sorted(ROUTES)}
    for row in rows:
        if row['id'] in ids or row['id'] in protected: raise ValueError('Duplicate or protected training ID')
        ids.add(row['id'])
        questions=decode(row['questions']);gold=decode(row['gold'])
        if set(questions)!={'route'} or set(questions['route']['criteria'])!=ROUTES: raise ValueError('Unexpected route schema')
        label=gold['route']['label']
        if label not in ROUTES: raise ValueError('Unexpected route label')
        counts[label]+=1
        clean.append({'id':row['id'],'state':row['state'],'questions':json.dumps(questions,ensure_ascii=False),'gold':json.dumps(gold,ensure_ascii=False),'n_questions':1,'route':label,'source':'synthetic-human-reviewed-phone-routing'})
    output.mkdir(parents=True,exist_ok=False)
    data=output/'data';data.mkdir()
    target=data/'train.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean))
    manifest={'source_train_sha256':expected,'export_train_sha256':sha(target),'rows':len(clean),'route_counts':counts,'published_splits':['train'],'excluded_splits':['validation','calibration','historical_test','fresh180_test'],'not_the_entire_combined_mixture':True,'no_network_or_upload':True,'license':'author must choose before upload','removed_fields':'reviewer identifiers, free-text review notes and UI metadata; approved training fields preserved'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--expected-train-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.input_dir,args.expected_train_sha256,args.output_dir),indent=2))


if __name__=='__main__':main()

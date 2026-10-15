#!/usr/bin/env python3
"""Publish aggregate-only matched comparison locally; never uploads raw cohorts."""
import hashlib
import json
from pathlib import Path
import re

from report_a32 import paired

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/decision-followups'
RUN = BASE/'laya-matched-20261015'
RELEASE = ROOT/'releases/reflex-1'


def read(path):
    return json.loads(path.read_text())


def main():
    if not (RUN/'complete.json').exists():
        raise RuntimeError('Wait for all four suites to complete')
    manifest = read(RUN/'run-manifest.json')
    protocol = read(RUN/'protocol.json')
    records = {}
    for name, slug in [('MOBILE_ROUTING9',None),('typed-decisions','typed_decisions'),
                       ('BANKING77','banking77'),('CLINC150','clinc150')]:
        laya = read(RUN/name/'summary.json')
        if slug:
            base = read(BASE/f'heldout-results-20261014/eval-base-s3407/{slug}.json')
            reflex = read(BASE/f'heldout-results-20261014/eval-combined-s3407/{slug}.json')
            assert base['sha256']==reflex['sha256']==laya['data_sha256']
            base_accuracy, reflex_accuracy = base['metrics']['accuracy'], reflex['metrics']['accuracy']
        else:
            base = read(BASE/'a32-fresh180/qwen-legacy-cached-fresh180/summary.json')
            reflex = read(BASE/'a32-fresh180/run-01/summary.json')['all_requests']
            base_accuracy, reflex_accuracy = base['accuracy'], reflex['accuracy']
        records[name] = dict(rows=laya['rows'], decisions=laya['decisions'],
                             base_accuracy=base_accuracy,reflex_accuracy=reflex_accuracy,
                             laya_accuracy=laya['accuracy'],laya_errors=laya['failures'],
                             laya_record_accuracy=laya['record_accuracy'],
                             data_sha256=laya['data_sha256'],
                             input_audit=laya['primary_audit'],
                             default_budget_audit=laya['shipped_budget_audit'],
                             laya_per_question_metrics=laya['by_question'])
    comparison = dict(complete=True,benchmark_names=list(records),results=records,
                       laya=dict(model=protocol['model'],revision=protocol['revision'],
                                 environment=manifest['environment'],profile=protocol['profile_kwargs'],
                                 batch_size=protocol['batch_size'],device=protocol['device']),
                       option_token_audit=read(RUN/'option-token-audit.json'),
                       mobile_paired=paired(BASE/'a32-fresh180/run-01',RUN/'MOBILE_ROUTING9'),
                       limitations='Scores use identical held-out cohorts and hard gold labels. Training exposure differs. Base general scores use an unseeded newly initialized decision head; base mobile uses the cached app-router. Reflex general is GPU PyTorch, Reflex mobile A32 Q4_K_M; Laya is Mac MPS. Timing is not compared across hardware. Laya English base is not its typed-decisions specialist. No test-based fitting.')
    (RELEASE/'laya-matched-summary.json').write_text(json.dumps(comparison,indent=2)+'\n')
    p = RELEASE/'HF_MODEL_CARD.md'
    text = p.read_text()
    for name,r in records.items():
        replacement = f"| {name} | {r['rows']} / {r['decisions']} | {100*r['base_accuracy']:.2f} | **{100*r['reflex_accuracy']:.2f}** | {100*r['laya_accuracy']:.2f} |"
        text = re.sub(r'^\| '+re.escape(name)+r' \|.*$', replacement, text, flags=re.M)
    text = text.replace('Laya scores are being measured on the same inputs with its English checkpoint, without additional training or test calibration.',
                        'Laya scores were measured on the same inputs with its pinned English checkpoint, without additional training or test calibration.')
    text = text.replace('Protocol and aggregate outcomes are supplied with the completed comparison.',
                        'All four Laya suites completed with '+str(sum(r['laya_errors'] for r in records.values()))+' request errors. [Protocol and aggregate outcomes](laya-matched-summary.json) include input-budget audits and the pinned runtime/checkpoint. Exact model revision: `'+protocol['revision']+'`; Laya SDK `'+manifest['environment']['laya']+'`.')
    p.write_text(text)
    for directory in [RUN/name for name in records]:
        inventory = dict(complete=True,files=[dict(file=f.name,bytes=f.stat().st_size,
                         sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(directory.iterdir()) if f.is_file() and f.name!='inventory.json'])
        (directory/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in r.items() if k.endswith('accuracy') or k=='laya_errors'} for name,r in records.items()},indent=2))


if __name__=='__main__':
    main()

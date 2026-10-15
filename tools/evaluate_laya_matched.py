#!/usr/bin/env python3
"""Evaluate a pinned shipped Laya checkpoint on the existing frozen cohorts."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import laya
import torch
from laya.common import build_sequence
from a32_queue import question_results, summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/decision-followups/laya-matched-20261015'


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, default=str)+'\n')


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def audit(agent, state, questions, max_len, head_max_len):
    result = {}
    for key, definition in questions.items():
        internal = agent._to_internal(definition)
        seq, markers, options, truncation = build_sequence(
            agent.tok, state, internal, max_len, head_max_len,
            return_stats=True, return_truncation_stats=True)
        result[key] = dict(tokens=len(seq), surviving_markers=len(markers),
                           options=options, state=truncation)
    return result


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    manifest = dict(protocol)
    manifest['environment'] = {name:importlib.metadata.version(name) for name in
                               ['laya','torch','transformers','huggingface_hub','safetensors']}
    manifest['platform'] = platform.platform()
    manifest['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()
    save(OUT/'run-manifest.json', manifest)
    start = time.monotonic()
    agent = laya.load(protocol['model'], revision=protocol['revision'], device='mps', fast=False)
    manifest['load_seconds'] = time.monotonic()-start
    manifest['loaded_config'] = agent.cfg
    manifest['loaded_revision'] = agent.revision
    manifest['dtype'] = str(agent.dtype)
    manifest['amp_min_rows'] = agent.mps_amp_min_rows
    save(OUT/'run-manifest.json', manifest)
    for name, data in protocol['files'].items():
        directory = OUT/name
        directory.mkdir(exist_ok=True)
        if (directory/'summary.json').exists():
            continue
        source = ROOT/data['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=data['sha256']:
            raise ValueError('Cohort SHA mismatch: '+name)
        cases = [json.loads(line) for line in source.read_text().splitlines()]
        path = directory/'predictions.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if [r['id'] for r in records] != [r['id'] for r in cases[:len(records)]]:
            raise ValueError('Resume prefix mismatch')
        cursor = len(records)
        while cursor<len(cases):
            questions = decode(cases[cursor]['questions'])
            batch = []
            while cursor+len(batch)<len(cases) and len(batch)<protocol['batch_size']:
                case = cases[cursor+len(batch)]
                if decode(case['questions'])!=questions:
                    break
                batch.append(case)
            audits = [audit(agent, case['state'], questions, **protocol['profile_kwargs']) for case in batch]
            default_audits = [audit(agent, case['state'], questions, 512,192) for case in batch]
            torch.mps.synchronize()
            begin = time.monotonic()
            error = None
            try:
                responses = agent.predict_batch([case['state'] for case in batch], questions,
                                                batch_size=protocol['batch_size'], **protocol['profile_kwargs'])
                torch.mps.synchronize()
                if len(responses)!=len(batch):
                    raise ValueError('Response count differs')
            except Exception as exc:
                error = repr(exc)
                responses = [None]*len(batch)
            elapsed = time.monotonic()-begin
            with path.open('a') as output:
                for case, response, primary_audit, default_audit in zip(batch,responses,audits,default_audits):
                    gold = decode(case['gold'])
                    result = dict(id=case['id'], state=case['state'],
                                  gold_labels={k:str(v['label']) for k,v in gold.items()},
                                  raw_response=response, input_audit=primary_audit,
                                  shipped_budget_audit=default_audit, batch_size=len(batch),
                                  batch_seconds=elapsed, latency_ms=1000*elapsed/len(batch),
                                  timing_note='Amortized batch time on Mac MPS, not per-request A32 latency')
                    try:
                        if error:
                            raise RuntimeError(error)
                        predictions, probabilities = question_results(response, questions)
                        result.update(predictions=predictions, probabilities=probabilities,
                                      correct=all(predictions.get(k)==v for k,v in result['gold_labels'].items()))
                    except Exception as exc:
                        result.update(error=repr(exc), correct=False)
                    records.append(result)
                    output.write(json.dumps(result)+'\n')
            cursor+=len(batch)
            save(directory/'progress.json', dict(completed=cursor,total=len(cases),errors=sum('error' in r for r in records)))
            print(name,cursor,'/',len(cases),'batch_s',round(elapsed,2),flush=True)
            if error:
                raise RuntimeError('Batch inference failure preserved; inspect before continuation: '+error)
        metrics = summary(records)
        metrics['timing_note'] = 'Latency fields are amortized batch runtime on Mac MPS; not comparable with A32 request latencies.'
        metrics['data_sha256'] = data['sha256']
        metrics['primary_audit'] = audit_summary(records,'input_audit')
        metrics['shipped_budget_audit'] = audit_summary(records,'shipped_budget_audit')
        save(directory/'summary.json', metrics)
        print('DONE',name,'accuracy',metrics['accuracy'],flush=True)
    save(OUT/'complete.json', dict(complete=True, suites=list(protocol['files'])))


def audit_summary(records, key):
    items = [item for row in records for item in row[key].values()]
    return dict(questions=len(items),state_truncated=sum(item['state']['truncated'] for item in items),
                missing_markers=sum(item['surviving_markers']!=item['options']['options'] for item in items),
                collapsed_options=sum(item['options']['options_distinct']<item['options']['options'] for item in items),
                capped_option_text=sum(item['options']['tokens_per_option'] is not None for item in items))


if __name__=='__main__':
    main()

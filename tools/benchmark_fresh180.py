#!/usr/bin/env python3
"""Frozen fresh-180 Decision API benchmark; ADB telemetry, raw outputs and parity.

Server startup is external. Run --help for required provenance arguments.
No relevance guard or prompt rewriting is applied. Never resumes partial runs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'experiments/decision-followups/fresh-phone-test-180'
EXPECTED_TEST = '2d18133e44a9f9db82fe80099b642f00cd6a12ebf643dc043d1f89f9ddfa209b'
EXPECTED_MODEL = 'abfea5f878331206f65e4117157984e14803b093b634c0ed52513c0723ef0699'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_inputs():
    assert sha(COHORT / 'test.jsonl') == EXPECTED_TEST, 'Frozen test hash changed'
    rows = [json.loads(x) for x in (COHORT / 'test.jsonl').read_text().splitlines()]
    schema = json.loads((COHORT / 'schema.json').read_text())
    refs = {x['id']: x for x in map(json.loads, (COHORT / 'results/predictions.jsonl').read_text().splitlines())}
    assert len(rows) == len(refs) == 180
    assert len({r['id'] for r in rows}) == 180
    for r in rows:
        assert json.loads(r['questions']) == schema
        assert refs[r['id']]['state'] == r['state']
        assert refs[r['id']]['gold'] == json.loads(r['gold'])['route']['label']
    return rows, schema, refs


def summarize(rows, labels):
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for r in rows:
        if r.get('prediction') in labels:
            matrix[r['gold']][r['prediction']] += 1
    f1s = []
    for label in labels:
        tp = matrix[label][label]
        support = sum(r['gold'] == label for r in rows)
        predicted = sum(m[label] for m in matrix.values())
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0)
    latencies = sorted(r['latency_ms'] for r in rows)
    paired = [r for r in rows if 'reference_agreement' in r]
    return {'total': len(rows), 'correct': sum(r.get('correct', False) for r in rows),
            'accuracy': sum(r.get('correct', False) for r in rows) / len(rows),
            'macro_f1': statistics.mean(f1s), 'failures': sum('error' in r for r in rows),
            'confusion_matrix': matrix,
            'mean_latency_ms': statistics.mean(latencies),
            'p50_latency_ms': statistics.median(latencies),
            'p95_latency_ms': latencies[math.ceil(.95 * len(latencies)) - 1],
            'reference_agreement': sum(r['reference_agreement'] for r in paired) / len(paired) if paired else None,
            'parity_valid_cases': len(paired)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--adb', default='adb')
    p.add_argument('--serial', default='R68T800LZ0H')
    p.add_argument('--url', default='http://127.0.0.1:8091')
    p.add_argument('--device-model', default='/sdcard/Download/models/reflex-1-combined-Q4_K_M.gguf')
    p.add_argument('--runtime-revision')
    p.add_argument('--launch-command', help='Exact server command, excluding secrets')
    p.add_argument('--pid', type=int)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    cases, schema, refs = load_inputs()
    if args.validate_only:
        print('Verified frozen SHA-256, all 180 IDs/states/gold labels and exact schema against reference.')
        return
    if not all([args.out, args.runtime_revision, args.launch_command, args.pid]):
        p.error('--out, --runtime-revision, --launch-command and --pid are required')
    def adb(*cmd):
        return subprocess.check_output([args.adb, '-s', args.serial, *cmd], text=True, timeout=30)
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2) + '\n')
    device_hash = adb('shell', 'sha256sum', args.device_model).split()[0]
    assert device_hash == EXPECTED_MODEL, 'Device model is not the selected combined checkpoint GGUF'
    def telemetry():
        return {name: adb('shell', *command) for name, command in {
            'battery': ['dumpsys', 'battery'], 'thermal': ['dumpsys', 'thermalservice'],
            'memory': ['dumpsys', 'meminfo', str(args.pid)],
            'process_status': ['cat', f'/proc/{args.pid}/status']}.items()}
    manifest = {'complete': False, 'test_sha256': EXPECTED_TEST,
          'model_sha256': device_hash, 'schema_sha256': sha(COHORT / 'schema.json'),
          'harness_sha256': sha(Path(__file__)), 'arguments': vars(args) | {'out': str(args.out)},
          'device_properties': adb('shell', 'getprop'), 'start_unix': time.time(),
          'protocol': 'Fixed order, first request separate, remaining sustained warm requests; no cooldown or guard; HTTP includes transport. Cold model load not measured.'}
    write('manifest.json', manifest)
    rows = []
    with (args.out / 'predictions.jsonl').open('w') as output:
        for index, case in enumerate(cases):
            ref = refs[case['id']]
            row = {'id': case['id'], 'state': case['state'], 'gold': ref['gold'],
                   'coverage': case['coverage'], 'before': telemetry()}
            start = time.monotonic()
            try:
                request = urllib.request.Request(args.url + '/v1/systemone',
                    json.dumps({'model': 'reflex-1', 'state': case['state'], 'questions': schema}).encode(),
                    {'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=300) as response:
                    row['raw_response'] = response.read().decode()
                row['latency_ms'] = 1000 * (time.monotonic() - start)
                result = json.loads(row['raw_response'])
                decision = result['answers']['route']
                probs = decision['probabilities']
                assert set(probs) == set(schema['route']['criteria'])
                assert all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
                assert abs(sum(probs.values()) - 1) < .01
                assert decision['choice'] in probs
                assert result['usage']['output_tokens'] == 0
                row.update(prediction=decision['choice'], correct=decision['choice'] == ref['gold'],
                           reference_agreement=decision['choice'] == ref['prediction'],
                           max_probability_drift=max(abs(probs[k] - ref['response']['probabilities'][k]) for k in probs))
            except Exception as e:
                row.setdefault('latency_ms', 1000 * (time.monotonic() - start))
                row.update(error=str(e), correct=False)
            output.write(json.dumps(row) + '\n')
            output.flush()  # Keep completed inference even if telemetry fails or device disconnects.
            row['after'] = telemetry()
            with (args.out / 'telemetry.jsonl').open('a') as trace:
                trace.write(json.dumps({'id': row['id'], 'after': row['after']}) + '\n')
            rows.append(row)
            write('progress.json', {'completed': len(rows), 'last_after': row['after']})
            print(index + 1, case['id'], row.get('prediction', row.get('error')), flush=True)
    labels = list(schema['route']['criteria'])
    write('summary.json', {'complete': True, 'all_requests': summarize(rows, labels),
                          'first_request_ms': rows[0]['latency_ms'],
                          'warm_requests': summarize(rows[1:], labels), 'end_unix': time.time()})
    manifest.update(complete=True, end_unix=time.time())
    write('manifest.json', manifest)
    write('telemetry.json', [{'id': r['id'], 'before': r['before'], 'after': r['after']} for r in rows])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compact read-only status for the sequential phone benchmark."""
import json
from pathlib import Path
import re

base=Path(__file__).resolve().parents[1]/'experiments/decision-followups/a32-fresh180'
progress=sorted(base.glob('*/progress.json'),key=lambda p:p.stat().st_mtime)
status={'completed_benchmark_jobs':len(list(base.glob('*/summary.json'))),'queue_complete':(base/'queue-complete.json').exists()}
if progress:
    p=progress[-1];r=json.loads(p.read_text());status.update(active=p.parent.name,completed_cases=r['completed'],total_cases=r.get('total',180))
    lines=(p.parent/'predictions.jsonl').read_text().splitlines()
    parsed=[]
    for line in lines:
        try:parsed.append(json.loads(line))
        except json.JSONDecodeError:pass
    if parsed:
        status['request_errors']=sum('error' in x for x in parsed)
        before=parsed[-1].get('before',{})
        m=re.search(r'temperature:\s*(\d+)',before.get('battery',''))
        if m:status['battery_c']=int(m[1])/10
        m=re.search(r'Thermal Status:\s*(\d+)',before.get('thermal',''))
        if m:status['android_thermal_status']=int(m[1])
queue_log=base/'queue-console.log'
if queue_log.exists():
    started=[line.split('START ',1)[1].strip() for line in queue_log.read_text().splitlines() if line.startswith('START ')]
    if started and not (base/started[-1]/'summary.json').exists() and status.get('active') != started[-1]:
        status.update(active=started[-1],phase='cooldown_or_startup',completed_cases=0)
        status.pop('battery_c',None);status.pop('android_thermal_status',None)
print(json.dumps(status))

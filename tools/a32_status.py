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
print(json.dumps(status))

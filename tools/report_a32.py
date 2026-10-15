#!/usr/bin/env python3
"""Build local report/card from completed outputs only. Never uploads."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/decision-followups/a32-fresh180'
JOBS=[('run-01','reflex-1','Fresh phone (180)'),('qwen-legacy-fresh180','Qwen app router (cache off)','Fresh phone (180)'),('qwen-legacy-cached-fresh180','Qwen app router (cache on)','Fresh phone (180)'),('beta-fresh180','reflex-1-beta','Fresh phone (180)'),('reflex-historical90','reflex-1','Historical phone (original 90)'),('qwen-historical90','Qwen app router (cache off)','Historical phone (original 90)'),('qwen-fresh180','Qwen schema JSON (cache off)','Fresh phone (180)'),('qwen-cached-fresh180','Qwen schema JSON (cache on)','Fresh phone (180)')]
for slug,label in [('reflex','reflex-1'),('beta','reflex-1-beta')]:
    for name,title in [('typed_decisions','typed-decisions'),('banking77','BANKING77'),('clinc150','CLINC150 in-domain')]:JOBS.append((slug+'-'+name,label,title))


def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def normalized(r):
    return r if 'gold_labels' in r else r|{'gold_labels':{'route':r['gold']},'predictions':{'route':r.get('prediction')}}


def resources(directory):
    records=rows(directory/'predictions.jsonl')
    telemetry=[r['before'] for r in records if 'before' in r]
    p=directory/'telemetry.jsonl'
    if p.exists():telemetry += [r['after'] for r in rows(p)]
    temps=[];ap=[];skin=[];pss=[];rss=[];thermal_status=[]
    for t in telemetry:
        m=re.search(r'temperature:\s*(\d+)',t.get('battery',''))
        if m:temps.append(int(m[1])/10)
        for name,values in [('AP',ap),('SKIN',skin)]:
            values += [float(x) for x in re.findall(r'mValue=([\d.]+), mType=\d+, mName='+name+r',',t.get('thermal',''))]
        thermal_status += [int(x) for x in re.findall(r'Thermal Status:\s*(\d+)',t.get('thermal',''))]
        pss += [int(x) for x in re.findall(r'TOTAL PSS:\s*(\d+)',t.get('memory',''))]
        rss += [int(x) for x in re.findall(r'VmHWM:\s*(\d+)',t.get('process_status',''))]
    return {'max_android_thermal_status':max(thermal_status) if thermal_status else None,'battery_c_min':min(temps) if temps else None,'battery_c_max':max(temps) if temps else None,'ap_c_max':max(ap) if ap else None,'skin_c_max':max(skin) if skin else None,'sampled_max_pss_kib':max(pss) if pss else None,'rss_high_water_kib':max(rss) if rss else None}





def usage_statistics(directory):
    inputs=[];outputs=[];cached=[];cache_denominators=[];evaluated=[];requested=[]
    for row in rows(directory/'predictions.jsonl'):
        if 'request_body' in row:requested.append(row['request_body'].get('cache_prompt'))
        if 'raw_response' not in row:continue
        try:response=json.loads(row['raw_response'])
        except json.JSONDecodeError:continue
        usage=response.get('usage') or {};timing=response.get('timings') or {}
        count=usage.get('prompt_tokens',usage.get('input_tokens'))
        out=usage.get('completion_tokens',usage.get('output_tokens'))
        if count is not None:inputs.append(count)
        if out is not None:outputs.append(out)
        details=usage.get('prompt_tokens_details') or {}
        cache=details.get('cached_tokens',timing.get('cache_n'))
        if cache is not None:
            cached.append(cache)
            if count is not None:cache_denominators.append(count)
        if timing.get('prompt_n') is not None:evaluated.append(timing['prompt_n'])
    avg=lambda values:statistics.mean(values) if values else None
    return {'requests_with_input_usage':len(inputs),'mean_reported_input_tokens':avg(inputs),'mean_output_tokens':avg(outputs),'requested_cache_values':sorted(set(requested)) if requested else [],'requests_with_cache_reporting':len(cached),'requests_reporting_nonzero_cached_tokens':sum(v>0 for v in cached),'mean_reported_cached_tokens':avg(cached),'cached_fraction_of_reported_input_tokens':sum(cached)/sum(cache_denominators) if cache_denominators and sum(cache_denominators) else None,'mean_server_prompt_evaluated_tokens':avg(evaluated),'note':'Measured server usage across all responses, including first request. Missing cache reporting means unknown, not zero. Requested flag alone does not prove reuse.'}


def grouped_results(directory):
    data=[normalized(r) for r in rows(directory/'predictions.jsonl')]
    output={}
    for facet in ['coverage','workflow']:
        groups={}
        for r in data:
            if facet not in r:continue
            cell=groups.setdefault(r[facet],{'rows':0,'decisions':0,'correct_decisions':0,'correct_records':0})
            cell['rows']+=1;cell['decisions']+=len(r['gold_labels'])
            cell['correct_decisions']+=sum(r.get('predictions',{}).get(k)==v for k,v in r['gold_labels'].items())
            cell['correct_records']+=all(r.get('predictions',{}).get(k)==v for k,v in r['gold_labels'].items())
        for cell in groups.values():
            cell['accuracy']=cell['correct_decisions']/cell['decisions']
            cell['record_accuracy']=cell['correct_records']/cell['rows']
        if groups:output[facet]=groups
    route={}
    if all(set(r['gold_labels'])=={'route'} for r in data):
        for r in data:
            gold=r['gold_labels']['route'];cell=route.setdefault(gold,{'support':0,'correct':0})
            cell['support']+=1;cell['correct']+=r.get('predictions',{}).get('route')==gold
        for cell in route.values():cell['recall']=cell['correct']/cell['support']
        output['route']=route
    return output


def calibration(directory):
    pairs=[];brier=[];nll=[]
    for original in rows(directory/'predictions.jsonl'):
        r=normalized(original)
        if 'error' in r:continue
        probs=r.get('probabilities')
        if probs is None and 'prediction' in r:
            probs={'route':json.loads(r['raw_response'])['answers']['route']['probabilities']}
        if probs is None:continue
        for name,gold in r['gold_labels'].items():
            pmf=probs[name];pred=r['predictions'][name]
            pairs.append((pmf[pred],int(pred==gold)))
            brier.append(sum((v-int(k==gold))**2 for k,v in pmf.items()))
            nll.append(-math.log(max(pmf[gold],1e-12)))
    if not pairs:return {'valid_decisions':0,'note':'No decision probability distribution available'}
    bins=[];ece=0
    for i in range(10):
        members=[(p,c) for p,c in pairs if min(int(p*10),9)==i]
        if not members:continue
        confidence=statistics.mean(p for p,c in members);accuracy=statistics.mean(c for p,c in members)
        ece+=len(members)/len(pairs)*abs(confidence-accuracy)
        bins.append({'bin':i,'n':len(members),'mean_selected_probability':confidence,'accuracy':accuracy})
    return {'valid_decisions':len(pairs),'ece_10_equal_width_bins':ece,'mean_multiclass_brier_sum':statistics.mean(brier),'hard_label_nll':statistics.mean(nll),'bins':bins,'definitions':'Explicit hard gold labels; Brier=sum over options, not divided by class count; NLL clipped at1e-12; failures excluded from calibration only, never accuracy.'}


def paired(a,b):
    left={r['id']:normalized(r) for r in rows(a/'predictions.jsonl')}
    right={r['id']:normalized(r) for r in rows(b/'predictions.jsonl')}
    if set(left)!=set(right):raise ValueError('Unmatched cohort IDs')
    deltas=[];better=worse=0
    for key,r in left.items():
        other=right[key]
        if r['state']!=other['state'] or r['gold_labels']!=other['gold_labels']:raise ValueError('Unmatched states/gold')
        x=all(r.get('predictions',{}).get(k)==v for k,v in r['gold_labels'].items())
        y=all(other.get('predictions',{}).get(k)==v for k,v in other['gold_labels'].items())
        better+=x and not y;worse+=y and not x;deltas.append(int(x)-int(y))
    n=better+worse
    p=min(1,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/(2**n)) if n else 1
    rng=random.Random(3407);boot=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(10000))
    return {'n':len(deltas),'reflex_only_correct':better,'baseline_only_correct':worse,'accuracy_difference':statistics.mean(deltas),'paired_bootstrap_95ci':[boot[249],boot[9749]],'exact_mcnemar_two_sided_p':p,'bootstrap_seed':3407,'bootstrap_replicates':10000}


def generate():
    records=[];lines=['| Model | A32 cohort | Cases / decisions | Accuracy | Macro-F1 | Warm p50 / p95 | Errors |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,label,title in JOBS:
        d=BASE/name;p=d/'summary.json'
        if not p.exists() or not json.loads(p.read_text()).get('complete'):
            progress=d/'progress.json';completed=json.loads(progress.read_text()).get('completed',0) if progress.exists() else 0
            lines.append(f'| {label} | {title} | {completed} completed | pending | pending | pending | pending |');continue
        s=json.loads(p.read_text());original=name=='run-01';total=s['all_requests'] if original else s;warm=s['warm_requests'] if original else s['warm']
        decision_count=total.get('decisions',total.get('total'))
        count=total.get('rows',total.get('total'))
        macro=total.get('macro_f1')
        if macro is None and len(total.get('by_question',{}))==1:macro=next(iter(total['by_question'].values()))['macro_f1']
        macro_text=f'{100*macro:.2f}%' if macro is not None else 'per-question in report'
        lines.append(f'| {label} | {title} | {count} / {decision_count} | {100*total["accuracy"]:.2f}% | {macro_text} | {warm["p50_latency_ms"]/1000:.2f} / {warm["p95_latency_ms"]/1000:.2f} s | {total["failures"]} |')
        records.append({'directory':name,'model':label,'cohort':title,'summary':s,'resources':resources(d),'token_usage':usage_statistics(d),'groups':grouped_results(d),'calibration':calibration(d),'predictions_sha256':hashlib.sha256((d/'predictions.jsonl').read_bytes()).hexdigest()})
    text='\n'.join(lines)
    resource_lines=['| Model | A32 cohort | Battery min–max | AP max | Android thermal status max | Sampled max PSS | RSS high-water |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        v=r['resources']
        display=lambda key,scale=1: f"{v[key]/scale:.2f}" if v[key] is not None else 'unavailable'
        resource_lines.append(f"| {r['model']} | {r['cohort']} | {display('battery_c_min')}–{display('battery_c_max')} °C | {display('ap_c_max')} °C | {display('max_android_thermal_status')} | {display('sampled_max_pss_kib',1024)} MiB | {display('rss_high_water_kib',1024)} MiB |")
    resource_table='\n'.join(resource_lines)
    comparisons={}
    for name in ['qwen-fresh180','qwen-cached-fresh180','qwen-legacy-fresh180','qwen-legacy-cached-fresh180','beta-fresh180']:
        if (BASE/name/'summary.json').exists() and (BASE/'run-01/summary.json').exists():comparisons[name]=paired(BASE/'run-01',BASE/name)
    report='# Samsung A32 benchmark results\n\nOnly completed jobs have scores; pending jobs are explicit. '+text+'\n\n'
    report=report.replace('explicit. |','explicit.\n\n|')
    report+='CPU-only 4 threads, Q4_K_M, runtime build11443, USB powered. Phone ctx2048, general ctx4096. Warm timing excludes first request and includes ADB/HTTP. General per-question accuracy and exact-record metrics are separate; CLINC excludes OOS. Raw Qwen uses fixed no-chat-template JSON-grammar treatment; not a claim about every Qwen prompting/runtime strategy.\n\n'
    report+=resource_table+'\n\nPSS maxima are sampled; RSS high-water is kernel-reported. USB power precludes an energy-saving claim.\n\n'
    report+='Reported token use and cache behaviour (all requests):\n\n```json\n'+json.dumps([{'directory':r['directory'],'usage':r['token_usage']} for r in records],indent=2)+'\n```\n\n'
    report+='Probability calibration (explicit hard labels, successful decisions only; failures still count in accuracy):\n\n```json\n'+json.dumps([{'directory':r['directory'],'metrics':r['calibration']} for r in records],indent=2)+'\n```\n\n'
    report+='Per-route, coverage and workflow results:\n\n```json\n'+json.dumps([{'directory':r['directory'],'groups':r['groups']} for r in records],indent=2)+'\n```\n\n'
    report+='Resource traces and measured sampled maxima:\n\n```json\n'+json.dumps([{k:v for k,v in r.items() if k in ['directory','resources']} for r in records],indent=2)+'\n```\n\nPaired fresh180 comparisons (reflex minus comparator):\n\n```json\n'+json.dumps(comparisons,indent=2)+'\n```\n'
    (BASE/'RESULTS.md').write_text(report)
    index={'completed_jobs':records,'paired_fresh180':comparisons,'complete':len(records)==len(JOBS)}
    (BASE/'results-index.json').write_text(json.dumps(index,indent=2)+'\n')
    public_summary=index|{'privacy':'Aggregate metrics only; no held-out requests, per-case responses, reviewer IDs or device-property dumps.', 'runtime':'Android CPU build11443 f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a; 4 inference/batch threads; Q4_K_M; phone ctx2048, general ctx4096; USB powered', 'cohort_caveat':'Fresh180 and original historical90 are distinct; historical PyTorch reference used reviewed88. Raw Qwen prompt/JSON grammar treatment is specifically documented, not all possible Qwen inference configurations.'}
    (ROOT/'releases/reflex-1/benchmark-summary.json').write_text(json.dumps(public_summary,indent=2)+'\n')
    card=ROOT/'releases/reflex-1/HF_MODEL_CARD.md'
    s=card.read_text();start='<!-- A32_RESULTS_START -->';end='<!-- A32_RESULTS_END -->'
    before,rest=s.split(start,1);_,after=rest.split(end,1)
    paired_text='\n\n'.join(f"Fresh180 vs {name}: accuracy difference {100*v['accuracy_difference']:+.2f} percentage points; paired bootstrap95% interval [{100*v['paired_bootstrap_95ci'][0]:+.2f}, {100*v['paired_bootstrap_95ci'][1]:+.2f}] points (10,000 resamples, seed3407). This interval concerns this reviewed synthetic cohort, not unrestricted real-world requests." for name,v in comparisons.items())
    card.write_text(before+start+'\n'+text+'\n\n'+resource_table+'\n\n'+paired_text+'\n\nCompleted runs only; partial counts are not scores. Warm timing excludes first request. General contexts4096 vs phone2048; see recorded protocol and per-run raw outputs.\n'+end+after)
    print('Report updated:',len(records),'/',len(JOBS),'completed')


if __name__=='__main__':generate()

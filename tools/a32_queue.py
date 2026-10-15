#!/usr/bin/env python3
"""Sequential Android benchmark queue. No uploads; resumes completed jobs only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/decision-followups'
OUT = BASE / 'a32-fresh180'
ADB = ['/Users/cbarkinozer/.cache/reflex-android/platform-tools/adb', '-s', 'R68T800LZ0H']
URL = 'http://127.0.0.1:8091'
RUNTIME = 'f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a'
MODELS = {
 'reflex-1': ('/sdcard/Download/models/reflex-1-combined-Q4_K_M.gguf','abfea5f878331206f65e4117157984e14803b093b634c0ed52513c0723ef0699'),
 'reflex-1-beta': ('/sdcard/Download/models/reflex-1-Q4_K_M.gguf','8480ccc295ac7379106f33ad76c978da5d9e8da0785e827786fe9d17f5a5beb5'),
 'qwen': ('/sdcard/Download/models/Qwen3.5-0.8B-Q4_K_M.gguf','bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517'),
}


def adb(*cmd):
    return subprocess.check_output(ADB + list(cmd), text=True, timeout=45)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def options(q):
    if q['type'] == 'noul': return ['false','true']
    if q['type'] == 'score': return [str(i) for i in range(len(q['criteria']))]
    return list(q['criteria'])


def grammar(questions):
    # JSON string literals encoded as GBNF string literals, never interpolate user text.
    lit = lambda text: json.dumps(json.dumps(text))
    fields = []
    rules = []
    for i,(name,q) in enumerate(questions.items()):
        fields.append(f'{lit(name)} ws ":" ws q{i}')
        rules.append(f'q{i} ::= ' + ' | '.join(lit(k) for k in options(q)))
    return 'root ::= "{" ws ' + ' ws "," ws '.join(fields) + ' ws "}"\n' + '\n'.join(rules) + '\nws ::= [ \\t\\n\\r]*\n'


def telemetry(pid):
    raw = {}
    for key,cmd in {'battery':['dumpsys','battery'], 'thermal':['dumpsys','thermalservice'],
                    'memory':['dumpsys','meminfo',str(pid)], 'process_status':['cat',f'/proc/{pid}/status']}.items():
        try: raw[key] = adb('shell',*cmd)
        except Exception as e: raw[key] = 'TELEMETRY_ERROR: ' + str(e)
    return raw


def temperature():
    b = adb('shell','dumpsys','battery')
    t = adb('shell','dumpsys','thermalservice')
    m = re.search(r'temperature:\s*(\d+)',b)
    status = re.search(r'Thermal Status:\s*(\d+)',t)
    if not m or not status: raise RuntimeError('Missing starting thermal readings')
    return int(m[1])/10,int(status[1])


def cool():
    start = time.time()
    while True:
        temp,status = temperature()
        if temp <= 30 and status == 0: return {'wait_seconds':time.time()-start,'temperature_c':temp,'thermal_status':status}
        if time.time()-start > 1800:
            # Preserve sustained-run evidence; avoid silently changing inference settings.
            if temp < 38 and status <= 1: return {'wait_seconds':time.time()-start,'temperature_c':temp,'thermal_status':status,'note':'30C target not reached after 30 min; below historical 38C gate'}
            raise RuntimeError('Unable to cool to historical thermal gate')
        time.sleep(10)


def start_server(label, context, directory):
    p=subprocess.run(ADB+['shell','pidof','llama-server'],capture_output=True,text=True)
    if p.returncode == 0:
        for pid in p.stdout.split(): adb('shell','kill','-INT',pid)
        time.sleep(2)
    cooldown=cool()
    path, expected=MODELS[label]
    actual=adb('shell','sha256sum',path).split()[0]
    if expected and actual != expected: raise RuntimeError('Incorrect model SHA '+label)
    command=f'cd /data/local/tmp/reflex-runtime && LD_LIBRARY_PATH=. ./llama-server -m {path} -c {context} -t 4 -tb 4 -ngl 0 --fit off -fa off --no-repack --host 127.0.0.1 --port 8091 --jinja'
    start=time.monotonic()
    launch=time.time()
    shellpid=adb('shell',f'nohup sh -c "{command}" > /data/local/tmp/reflex-runtime/overnight-server.log 2>&1 < /dev/null & echo $!').strip()
    adb('forward','tcp:8091','tcp:8091')
    for _ in range(90):
        try:
            with urllib.request.urlopen(URL+'/health',timeout=1) as r: ready=json.load(r).get('status')=='ok'
            if ready: break
        except Exception: pass
        time.sleep(1)
    else: raise RuntimeError('Server did not become ready')
    pid=int(adb('shell','pidof','llama-server').strip())
    record={'label':label,'model_sha256':actual,'runtime_revision':RUNTIME,'launch_command':command,'pid':pid,'shell_pid':shellpid,'launch_unix':launch,'ready_seconds':time.monotonic()-start,'cooldown':cooldown,'cache_flushed':False,'device_properties':adb('shell','getprop')}
    save(directory/'startup.json',record)
    return pid,record


def question_results(response, questions):
    predicted={};probs={}
    for name,q in questions.items():
        result=response['answers'][name]
        choices=options(q)
        if q['type']=='noul':
            prob=result.get('probabilities')
            if prob is None: prob={'false':1-result['noul'],'true':result['noul']}
        else: prob=result['probabilities']
        if set(prob)!=set(choices) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or v>1 for v in prob.values()) or abs(sum(prob.values())-1)>.01:
            raise ValueError('Invalid probability options/normalization: '+name)
        probs[name]=prob
        # Same discrete argmax scoring used by the saved PyTorch evaluator, not rounded expected score.
        predicted[name]=result['choice'] if q['type']=='choice' else max(prob,key=prob.get)
        if predicted[name] not in choices: raise ValueError('Invalid choice')
    if response['usage']['output_tokens']!=0: raise ValueError('Unexpected generated tokens')
    return predicted,probs


def summary(rows):
    flat=[(name,gold,r.get('predictions',{}).get(name)) for r in rows for name,gold in r['gold_labels'].items()]
    by_name={}
    for name in sorted({x[0] for x in flat}):
        pairs=[(g,p) for n,g,p in flat if n==name]
        labels=sorted({g for g,p in pairs})
        counts={label:{'support':sum(g==label for g,p in pairs),'predicted':sum(p==label for g,p in pairs),'tp':sum(g==label and p==label for g,p in pairs)} for label in labels}
        f1s=[2*x['tp']/(x['support']+x['predicted']) if x['support']+x['predicted'] else 0 for x in counts.values()]
        by_name[name]={'accuracy':sum(g==p for g,p in pairs)/len(pairs),'macro_f1':statistics.mean(f1s),'per_label':counts}
    lat=sorted(r['latency_ms'] for r in rows)
    return {'complete':True,'rows':len(rows),'decisions':len(flat),'accuracy':sum(g==p for n,g,p in flat)/len(flat),'record_accuracy':sum(all(r.get('predictions',{}).get(k)==v for k,v in r['gold_labels'].items()) for r in rows)/len(rows),'failures':sum('error' in r for r in rows),'mean_latency_ms':statistics.mean(lat),'p50_latency_ms':statistics.median(lat),'p95_latency_ms':lat[math.ceil(.95*len(lat))-1],'by_question':by_name}


def run_suite(label, mode, data, directory, reference=None):
    directory.mkdir(parents=True,exist_ok=True)
    old_manifest=directory/'manifest.json'
    if old_manifest.exists():
        previous=json.loads(old_manifest.read_text())
        if previous['data_sha256'] != sha(data) or previous['mode'] != mode or previous['startup']['label'] != label:
            raise RuntimeError('Resume inputs differ from saved manifest')
    done=directory/'summary.json'
    if done.exists() and json.loads(done.read_text()).get('complete'): return
    rows=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    context=2048 if 'phone' in str(data) else 4096
    pid,startup=start_server(label,context,directory)
    manifest={'data_path':str(data.relative_to(ROOT)) if data.is_relative_to(ROOT) else str(data),'data_sha256':sha(data),'rows':len(rows),'mode':mode,'startup':startup,'harness_sha256':sha(Path(__file__)),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'start_unix':time.time(),'complete':False,'protocol':'Sequential fixed-order sustained requests; no relevance guard or per-request cooldown; first request separated; USB powered; HTTP includes ADB transport. Raw baseline uses identical semantic state and question descriptions with JSON grammar, no chat template. General decision scores use gold label not soft-target argmax.'}
    save(directory/'manifest.json',manifest)
    existing=[]
    prediction_path=directory/'predictions.jsonl'
    if prediction_path.exists():
        existing=[json.loads(l) for l in prediction_path.read_text().splitlines() if l.strip()]
        if [r['id'] for r in existing]!=[r['id'] for r in rows[:len(existing)]]: raise RuntimeError('Resume prefix mismatch')
        save(directory/'resume.json',{'resumed_rows':len(existing),'new_startup':startup,'warning':'Restart changes timing and thermal sequence; report separately'})
    results=existing
    refs={} if reference is None else {r['id']:r for r in map(json.loads,reference.read_text().splitlines())}
    with prediction_path.open('a') as output:
        for case in rows[len(existing):]:
            questions=decode(case['questions']);gold=decode(case['gold'])
            result={'id':case['id'],'state':case['state'],'gold_labels':{k:str(v['label']) for k,v in gold.items()},'before':telemetry(pid)}
            if 'coverage' in case: result['coverage']=case['coverage']
            if 'workflow' in case: result['workflow']=case['workflow']
            if 'source' in case: result['source']=case['source']
            if mode=='decision':
                endpoint='/v1/systemone';body={'model':label,'state':case['state'],'questions':questions}
            else:
                endpoint='/v1/completions'
                prompt='Select the best answer for every question using the input. Return one JSON object mapping each question name to its option key. Return only JSON. For yes/no use "true" or "false"; for scores use the level index as a string.\nQuestions:\n'+json.dumps(questions,ensure_ascii=False)+'\nInput:\n'+case['state']+'\nDecision:\n'
                body={'prompt':prompt,'grammar':grammar(questions),'temperature':.3,'seed':3407,'max_tokens':256,'cache_prompt':False}
                result['request_body']=body
            begin=time.monotonic()
            try:
                req=urllib.request.Request(URL+endpoint,json.dumps(body).encode(),{'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=300) as r: result['raw_response']=r.read().decode()
                result['latency_ms']=1000*(time.monotonic()-begin)
                response=json.loads(result['raw_response'])
                if mode=='decision': predictions,prob=question_results(response,questions);result['probabilities']=prob
                else:
                    predictions=json.loads(response['choices'][0]['text'])
                    if set(predictions)!=set(questions) or any(predictions[k] not in options(q) for k,q in questions.items()): raise ValueError('Invalid baseline result')
                    if response.get('truncated') or response.get('stop_type')=='limit': raise ValueError('Baseline truncation/token budget limit')
                result['predictions']=predictions
                result['correct']=all(predictions.get(k)==v for k,v in result['gold_labels'].items())
                if case['id'] in refs:
                    ref=refs[case['id']];result['reference_agreement']=predictions.get('route')==ref['prediction']
                    if mode=='decision':result['max_probability_drift']=max(abs(prob['route'][k]-ref['response']['probabilities'][k]) for k in prob['route'])
            except Exception as e:
                result['error']=str(e);result['correct']=False
                if isinstance(e,urllib.error.HTTPError):result['raw_error_body']=e.read().decode(errors='replace')
                result.setdefault('latency_ms',1000*(time.monotonic()-begin))
            output.write(json.dumps(result)+'\n');output.flush()
            result['after']=telemetry(pid)
            with (directory/'telemetry.jsonl').open('a') as trace:trace.write(json.dumps({'id':case['id'],'after':result['after']})+'\n')
            results.append(result)
            save(directory/'progress.json',{'completed':len(results),'total':len(rows),'correct_records':sum(r['correct'] for r in results),'last_id':case['id'],'last_latency_ms':result['latency_ms'],'updated_unix':time.time()})
            print(directory.name,len(results),'/',len(rows),case['id'],result.get('predictions',result.get('error')),round(result['latency_ms']/1000,2),flush=True)
    report=summary(results);report['warm']=summary(results[1:]);report['first_request_ms']=results[0]['latency_ms']
    save(directory/'summary.json',report)
    save(directory/'manifest.json',manifest|{'complete':True,'end_unix':time.time()})
    subprocess.run(ADB+['pull','/data/local/tmp/reflex-runtime/overnight-server.log',str(directory/'server.log')],check=True)
    adb('shell','kill','-INT',str(pid))


def queue():
    OUT.mkdir(parents=True,exist_ok=True)
    ref=BASE/'fresh-phone-test-180/results/predictions.jsonl'
    fresh=BASE/'fresh-phone-test-180/test.jsonl'
    historical=Path('/Users/cbarkinozer/Documents/GitHub/pocket-agents-lab/data/phone_routing/test.jsonl')
    general=BASE/'heldout-results-20261014/frozen-general-tests'
    jobs=[('qwen','raw',fresh,'qwen-fresh180',ref),
          ('reflex-1-beta','decision',fresh,'beta-fresh180',ref),
          ('reflex-1','decision',historical,'reflex-historical90',None),
          ('qwen','raw',historical,'qwen-historical90',None)]
    # Full frozen general cohorts, not score-driven subsets. Both decision versions.
    for label,slug in [('reflex-1','reflex'),('reflex-1-beta','beta')]:
        for suite in ['typed_decisions','banking77','clinc150']:
            jobs.append((label,'decision',general/(suite+'.jsonl'),slug+'-'+suite,None))
    plan=[{'model':m,'mode':mode,'data':str(data),'out':str(OUT/name),'data_sha256':sha(data)} for m,mode,data,name,refp in jobs]
    save(OUT/'queue-plan.json',{'created_unix':time.time(),'jobs':plan,'no_uploads':True,'scope':'Matched fresh and historical phone comparisons plus all three frozen general suites for both decision versions. General suites can exceed overnight budget; no silent downsampling.'})
    # Wait for the original frozen-180 run; never run two phone jobs concurrently.
    print('Waiting for active reflex-1 fresh180 run to finish',flush=True)
    while True:
        p=OUT/'run-01/summary.json'
        if p.exists() and json.loads(p.read_text()).get('complete'):break
        progress=OUT/'run-01/progress.json'
        if progress.exists(): print('reflex-1 fresh180',json.loads(progress.read_text()).get('completed'),'/180',flush=True)
        if subprocess.run(['pgrep','-f','tools/benchmark_fresh180.py'],stdout=subprocess.DEVNULL).returncode:
            raise RuntimeError('Original run stopped without summary; inspect and resume explicitly')
        time.sleep(30)
    subprocess.run(ADB+['pull','/data/local/tmp/reflex-runtime/reflex-combined-server.log',str(OUT/'run-01/server.log')],check=True)
    for label,mode,data,name,refp in jobs:
        print('START',name,flush=True)
        run_suite(label,mode,data,OUT/name,refp)
        print('DONE',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'tools/report_a32.py')],check=True)
    save(OUT/'queue-complete.json',{'complete':True,'end_unix':time.time(),'jobs':len(jobs)})


if __name__=='__main__':
    queue()

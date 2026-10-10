"""CPU GGUF load/API smoke, not a full accuracy or phone benchmark."""
import json,subprocess,time,urllib.request
from pathlib import Path
root=Path('/workspace/reflex-decision');out=root/'runs/reflex-1-gguf'
request=json.loads((root/'decision-request.json').read_text());request['model']='reflex-1'
reference=json.loads((root/'runs/inference-demo/merged-response.json').read_text())['answers']
results={}
for quant in ['Q8_0','Q4_K_M']:
    log=(out/f'server-{quant}.log').open('w')
    process=subprocess.Popen([str(root/'llama.cpp/build/bin/llama-server'),'-m',str(out/f'model-{quant}.gguf'),'--port','8091','--host','127.0.0.1','-c','4096','-t','8','-ngl','0'],stdout=log,stderr=subprocess.STDOUT)
    try:
        ready=False
        for _ in range(180):
            if process.poll() is not None:raise RuntimeError(f'{quant} server failed: see log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8091/health',timeout=1) as response:
                    if response.status==200:ready=True;break
            except Exception:time.sleep(1)
        if not ready:raise RuntimeError('Server readiness timed out')
        start=time.perf_counter()
        req=urllib.request.Request('http://127.0.0.1:8091/v1/systemone',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as response:answer=json.load(response)
        elapsed=time.perf_counter()-start
        assert set(answer['answers'])==set(reference)
        for key,v in answer['answers'].items():
            assert v['type']==reference[key]['type']
            if 'probabilities' in v:assert abs(sum(v['probabilities'].values())-1)<.002
        def chosen(v):
            if v['type']=='choice':return v['choice']
            if v['type']=='noul':return v['noul']>=.5
            return max(v['probabilities'],key=v['probabilities'].get)
        results[quant]={'response':answer,'cpu_request_seconds':elapsed,'changed_answers':{key:{'pytorch':reference[key]['answer'],'gguf':chosen(v)} for key,v in answer['answers'].items() if str(chosen(v)).lower()!=str(reference[key]['answer']).lower()},'scope':'single structured request, CPU load/API smoke; not full benchmark or Android validation'}
        (out/'api-validation.json').write_text(json.dumps(results,indent=2))
        print(quant,'API validated',elapsed,flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()

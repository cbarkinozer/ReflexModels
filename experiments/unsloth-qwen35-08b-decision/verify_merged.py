"""Reload merged weights and exercise the structured predictor without generation."""
import os,json,time
from pathlib import Path
os.environ.update(HF_HOME='/workspace/reflex-decision/cache/huggingface',HF_HUB_OFFLINE='1',UNSLOTH_DISABLE_STATISTICS='1',UNSLOTH_COMPILE_DISABLE='1',WANDB_MODE='disabled')
from unsloth import FastDecisionModel
root=Path('/workspace/reflex-decision');out=root/'runs/inference-demo'
model,tokenizer=FastDecisionModel.from_pretrained(str(root/'runs/notebook-60steps/merged'),load_in_4bit=True,local_files_only=True,max_seq_length=4096)
FastDecisionModel.for_inference(model)
def forbid(*args,**kwargs):raise RuntimeError('Generation forbidden')
model.generate=forbid
request=json.loads((out/'request.json').read_text());start=time.perf_counter();answers=FastDecisionModel.predict(model,tokenizer,request['state'],request['questions'])
previous=json.loads((out/'responses.json').read_text())[-1]['answers']
for v in answers.values():assert abs(sum(v['probabilities'].values())-1)<.001
result={'answers':answers,'latency_seconds':time.perf_counter()-start,'loaded_merged_in_4bit':True,'text_generation_forbidden':True,'changed_answers':{k:{'adapter':previous[k]['answer'],'merged':v['answer']} for k,v in answers.items() if previous[k]['answer']!=v['answer']},'maximum_probability_delta':max(abs(p-previous[k]['probabilities'][option]) for k,v in answers.items() for option,p in v['probabilities'].items())}
(out/'merged-response.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

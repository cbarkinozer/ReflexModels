"""Gold-label benchmark evaluation; deterministic class-balanced official test subsets."""
import os, json, random, hashlib, csv, io, gc, time, math
from pathlib import Path
os.environ.update(HF_HOME='/workspace/reflex-decision/cache/huggingface', UNSLOTH_DISABLE_STATISTICS='1', UNSLOTH_COMPILE_DISABLE='1', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
from unsloth import FastDecisionModel
import torch, requests
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
ROOT=Path('/workspace/reflex-decision'); OUT=ROOT/'runs/balanced-benchmarks'; OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'runs/notebook-60steps/manifest.json').read_text())
seed=3407
revision=requests.get('https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/master',timeout=60).json()['sha']
url=f'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data/test.csv'
response=requests.get(url,timeout=60); response.raise_for_status(); (OUT/'banking77-test.csv').write_bytes(response.content)
bank=[{'text':r['text'],'label':r['category'],'source_index':i} for i,r in enumerate(csv.DictReader(io.StringIO(response.text)))]
clinc_rev=HfApi().dataset_info('clinc/clinc_oos').sha
clinc=load_dataset('clinc/clinc_oos','plus',split='test',revision=clinc_rev)
names=clinc.features['intent'].names
clinc_rows=[{'text':r['text'],'label':names[r['intent']],'source_index':i} for i,r in enumerate(clinc) if names[r['intent']]!='oos']
train=load_dataset('LocalLLaMA/typed-decisions','all',split='train',revision=manifest['dataset_revision'])
normalize=lambda s:' '.join(s.casefold().split())
training_states={normalize(r['state']) for r in train}
def make(rows,n):
 groups={}
 for r in rows:
  if normalize(r['text']) not in training_states: groups.setdefault(r['label'],[]).append(r)
 rng=random.Random(seed); selected=[]
 for label in sorted(groups): selected.extend(rng.sample(groups[label],n))
 rng.shuffle(selected); labels=sorted(groups)
 question={'intent':{'type':'choice','instructions':'Which intent best matches the user request? Select exactly one intent.','criteria':{k:k.replace('_',' ').replace('?','') for k in labels}}}
 items=[]
 for r in selected:
  items.append({'state':r['text'],'questions':json.dumps(question),'gold':json.dumps({'intent':{'type':'choice','label':r['label'],'probabilities':{k:float(k==r['label']) for k in labels}}})})
 return items,selected,{'classes':len(labels),'per_class':n,'records':len(items),'options':'all dataset intent classes','excluded_oos':True,'state_overlap_with_typed_training':0}
typed=load_dataset('LocalLLaMA/typed-decisions','all',split='test',revision=manifest['dataset_revision'])
workflows={}
for i,r in enumerate(typed): workflows.setdefault(r['workflow'],[]).append(i)
assert len({len(v) for v in workflows.values()})==1
sets={}
for name,rows,n in [('banking77',bank,6),('clinc150',clinc_rows,3)]:
 items,selection,meta=make(rows,n);sets[name]=Dataset.from_list(items)
 (OUT/f'{name}-selection.json').write_text(json.dumps({'metadata':meta,'selection':selection,'decision_rows':items},indent=2))
sets['typed-decisions']=typed
(OUT/'config.json').write_text(json.dumps({'seed':seed,'banking77_github_revision':revision,'banking77_test_sha256':hashlib.sha256(response.content).hexdigest(),'clinc_revision':clinc_rev,'model_revision':manifest['model_revision'],'typed_revision':manifest['dataset_revision'],'typed_workflow_counts':{k:len(v) for k,v in workflows.items()},'calibration_refitted':False,'evaluation_max_seq_length':4096,'training_max_seq_length':2048,'note':'All 77/150 intent candidates, balanced test samples; not claimed identical to blog option construction.'},indent=2))
for stage,path in [('before','unsloth/Qwen3.5-0.8B'),('trained',str(ROOT/'runs/notebook-60steps/adapter'))]:
 kwargs={'revision':manifest['model_revision']} if stage=='before' else {'local_files_only':True}
 torch.manual_seed(seed)
 model,tokenizer=FastDecisionModel.from_pretrained(path,load_in_4bit=True,max_seq_length=4096,**kwargs)
 for name,data in sets.items():
  started=time.time();items,report=FastDecisionModel.build_dataset(data,tokenizer,model)
  assert report['skipped']==0 and items, report
  metrics=FastDecisionModel.evaluate(model,tokenizer,items)
  assert all(math.isfinite(metrics[k]) for k in ('accuracy','ece','loss')), metrics
  result={'stage':stage,'dataset':name,'metrics':metrics,'build_report':report,'duration_seconds':time.time()-started,'calibration_refitted':False}
  (OUT/f'{stage}-{name}.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
 del model,tokenizer;gc.collect();torch.cuda.empty_cache()
(OUT/'complete.json').write_text(json.dumps({'completed':True}))

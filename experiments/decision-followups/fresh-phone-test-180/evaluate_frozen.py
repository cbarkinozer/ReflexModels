"""Evaluate the frozen fresh-phone test; retain every serving prediction."""
import argparse,collections,hashlib,json,math,os,time
from pathlib import Path
os.environ['WANDB_MODE']='disabled'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True)
 a=p.parse_args();m=json.loads(a.manifest.read_text())
 if sha(a.test)!=m['test_sha256']:raise ValueError('Frozen test hash mismatch')
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True)
 from unsloth import FastDecisionModel
 import torch
 from datasets import Dataset
 torch.manual_seed(3407)
 start=time.time();rows=[json.loads(s) for s in a.test.read_text(encoding='utf-8').splitlines() if s.strip()]
 model,tokenizer=FastDecisionModel.from_pretrained(model_name=a.model,max_seq_length=4096,dtype=None,load_in_4bit=True)
 model.eval()
 items,report=FastDecisionModel.build_dataset(Dataset.from_list(rows),tokenizer,model)
 if len(items)!=180 or report.get('skipped',0) or report.get('truncated',0):raise ValueError(report)
 metrics=FastDecisionModel.evaluate(model,tokenizer,items,batch_size=8)
 labels=list(json.loads(rows[0]['questions'])['route']['criteria']);conf={k:{j:0 for j in labels} for k in labels};records=[]
 dest=a.out/'predictions.jsonl'
 with dest.open('w',encoding='utf-8') as handle,torch.inference_mode():
  for i,r in enumerate(rows):
   response=FastDecisionModel.predict(model,tokenizer,r['state'],json.loads(r['questions']))['route']
   prediction=response['answer'];gold=r['route__label']
   if prediction not in labels:raise ValueError(response)
   conf[gold][prediction]+=1
   rec={'id':r['id'],'state':r['state'],'gold':gold,'prediction':prediction,'correct':gold==prediction,'coverage':r['coverage'],'response':response}
   records.append(rec);handle.write(json.dumps(rec,ensure_ascii=False)+'\n');handle.flush()
   if (i+1)%20==0:print('Serving predictions',i+1,'/180',flush=True)
 correct=sum(r['correct'] for r in records);accuracy=correct/len(records)
 if abs(accuracy-float(metrics['accuracy']))>1e-6:raise ValueError('Batch and serving accuracy differ; inspect saved predictions before reporting')
 per={}
 for label in labels:
  tp=conf[label][label];support=sum(conf[label].values());predicted=sum(conf[k][label] for k in labels)
  precision=tp/predicted if predicted else 0;recall=tp/support
  per[label]={'support':support,'correct':tp,'accuracy':recall,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0}
 n=len(records);z=1.959963984540054;den=1+z*z/n;center=(accuracy+z*z/(2*n))/den;half=z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/den
 summary={'model':a.model,'source_commit':a.source_commit,'test_sha256':sha(a.test),'rows':n,'correct':correct,'accuracy':accuracy,'macro_f1':sum(v['f1'] for v in per.values())/len(labels),'wilson_95_accuracy':[center-half,center+half],'native_metrics':metrics,'build_report':report,'per_route':per,'confusion_matrix':conf,'per_coverage':{k:{'rows':sum(r['coverage']==k for r in records),'correct':sum(r['coverage']==k and r['correct'] for r in records)} for k in ['everyday','indirect','boundary']},'elapsed_seconds':time.time()-start,'max_seq_length':4096,'batch_size':8,'seed':3407,'predictions_sha256':sha(dest),'selection':'Fixed checkpoint 100 selected before this test; no tuning on results'}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 (a.out/'errors.json').write_text(json.dumps([r for r in records if not r['correct']],indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

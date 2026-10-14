"""Import/export the fresh phone test review without modifying existing datasets."""
import argparse,datetime,hashlib,json
from pathlib import Path
from argilla_phone_review import read_env,ROUTES
ROOT=Path(__file__).resolve().parent/'fresh-phone-test-180'
NAME='phone-routing-fresh-test-180-review-20261014'
GUIDELINES='''# Fresh phone test review
Choose the route from the meaning of the request. The preselected route is a synthetic author proposal, not a model prediction or a verified label. Check it and correct it when needed. These synthetic candidates are not yet a sealed test.

answer: knowledge, calculation, writing, or clarification when no stable tool intent is inferable.
get_battery_info: this handset's live charge, charging, cell condition, temperature or runtime.
get_device_info: device identity, installed software, hardware specifications/capabilities (including advertised battery capacity and flash hardware).
get_storage_info: live disk capacity, usage or mounted/writable volumes.
get_media_info: current/recent playback, metadata, settings and media availability.
phone_health_check: overall condition/readiness or multiple live device categories.
save_note: explicitly preserve new information supplied in the request.
search_notes: retrieve an existing note from the notes system.
search_local_files: find a filesystem object by name, contents, type, time or origin, including exported notes archives.

Label by intended capabilities, including plausible future features. Ignore today's implementation limitations. Battery-only remains battery-specific. Distinguish saved-note retrieval from finding an exported file. General explanations do not need live phone data.

Select accept if the request is clear and usable, or needs_revision/exclude/uncertain. You may supply a replacement request and a reason. Do not force an ambiguous example into a route merely to fill a quota. Submit each review. After review we will audit labels, restore balance if necessary and freeze a hash. No training or model evaluation occurs here.
'''
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('mode',choices=['import','export','suggest'])
 p.add_argument('--credential-env',type=Path,required=True)
 p.add_argument('--url',default='http://127.0.0.1:7900')
 p.add_argument('--workspace',default='reflexmodels')
 args=p.parse_args()
 import argilla as rg
 client=rg.Argilla(api_url=args.url,api_key=read_env(args.credential_env)['ARGILLA_API_KEY'])
 rows=[json.loads(s) for s in (ROOT/'candidates.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
 dataset=client.datasets(name=NAME,workspace=args.workspace)
 if args.mode=='suggest':
  if dataset is None:raise ValueError('Review dataset missing')
  candidates={r['id']:r for r in rows}; pending=[];before={}
  for record in dataset.records:
   rid=str(record.id)
   before[rid]=[v.model_dump(mode='json') for v in record.responses.api_models()]
   if before[rid]:continue
   pending.append(rg.Record(id=rid,fields=dict(record.fields),suggestions=[rg.Suggestion(question_name='correct_route',value=candidates[rid]['candidate_route'],agent='synthetic-author-proposal')]))
  if pending:dataset.records.log(pending)
  after={str(r.id):[v.model_dump(mode='json') for v in r.responses.api_models()] for r in dataset.records}
  if before!=after:raise ValueError('Human responses changed unexpectedly')
  for r in dataset.records:
   if not before[str(r.id)] and not any(s.question_name=='correct_route' and s.value==candidates[str(r.id)]['candidate_route'] for s in r.suggestions):raise ValueError('Suggestion verification failed')
  links=ROOT/'links.json'; data=json.loads(links.read_text());data['route_suggestions']=True;data['suggestion_source']='synthetic author; not model prediction';links.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
  print(json.dumps({'suggestions_added':len(pending),'submitted_reviews_preserved':sum(bool(v) for v in before.values())}));return
 if args.mode=='export':
  if dataset is None:raise ValueError('Review dataset missing')
  users={str(u.id):u.username for u in client.users}
  records=[{'id':str(r.id),'fields':dict(r.fields),'responses':[v.model_dump(mode='json') for v in r.responses.api_models()]} for r in dataset.records]
  snapshot={'exported_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':NAME,'users':users,'records':records,'status':'review_only_not_frozen'}
  dest=ROOT/'review-snapshot.json';temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(dest)
  print(json.dumps({'exported':len(records),'reviewed':sum(bool(r['responses']) for r in records),'snapshot':str(dest)}));return
 if len(rows)!=180 or len({r['id'] for r in rows})!=180:raise ValueError('Expected 180 unique candidates')
 if client.workspaces(name=args.workspace) is None:rg.Workspace(name=args.workspace,client=client).create()
 if dataset is None:
  settings=rg.Settings(guidelines=GUIDELINES,fields=[rg.TextField(name='request',title='User request')],questions=[
   rg.LabelQuestion(name='correct_route',title='Correct tool / route',labels=ROUTES,required=True),
   rg.LabelQuestion(name='review_decision',title='Is this example ready for the test?',labels=['accept','needs_revision','exclude','uncertain'],required=True),
   rg.TextQuestion(name='replacement_request',title='Replacement wording (optional)',required=False),
   rg.TextQuestion(name='review_notes',title='Reason or ambiguity (optional)',required=False)],metadata=[rg.TermsMetadataProperty(name='coverage',options=['everyday','indirect','boundary'])])
  dataset=rg.Dataset(name=NAME,workspace=args.workspace,settings=settings,client=client).create()
 existing={str(r.id):dict(r.fields) for r in dataset.records};expected={r['id']:{'request':r['state']} for r in rows}
 for k,v in existing.items():
  if k not in expected or v!=expected[k]:raise ValueError('Existing review differs; preserved')
 records=[rg.Record(id=r['id'],fields=expected[r['id']],metadata={'coverage':r['coverage']},suggestions=[rg.Suggestion(question_name='correct_route',value=r['candidate_route'],agent='synthetic-author-proposal')]) for r in rows if r['id'] not in existing]
 if records:dataset.records.log(records)
 actual={str(r.id):dict(r.fields) for r in dataset.records}
 if actual!=expected:raise ValueError('Import verification failed')
 result={'name':NAME,'id':str(dataset.id),'records':180,'url':f'{args.url}/dataset/{dataset.id}/annotation-mode','candidate_sha256':hashlib.sha256((ROOT/'candidates.jsonl').read_bytes()).hexdigest(),'status':'unreviewed_candidates_not_sealed','route_suggestions':True,'model_evaluated':False}
 (ROOT/'links.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

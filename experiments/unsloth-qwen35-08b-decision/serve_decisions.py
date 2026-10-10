"""Local HTTP adapter for our trained Qwen decision head; not Unsloth Studio itself."""
import os,json,argparse,time
from http.server import HTTPServer,BaseHTTPRequestHandler
os.environ.update(HF_HOME='/workspace/reflex-decision/cache/huggingface', HF_HUB_OFFLINE='1', UNSLOTH_DISABLE_STATISTICS='1', UNSLOTH_COMPILE_DISABLE='1', WANDB_MODE='disabled')
from unsloth import FastDecisionModel
p=argparse.ArgumentParser();p.add_argument('--model',default='/workspace/reflex-decision/runs/notebook-60steps/adapter');p.add_argument('--port',type=int,default=8080);a=p.parse_args()
model,tokenizer=FastDecisionModel.from_pretrained(a.model,load_in_4bit=True,local_files_only=True,max_seq_length=4096)
FastDecisionModel.for_inference(model)
def forbid_generation(*args,**kwargs):raise RuntimeError('Text generation is forbidden in this decision service')
model.generate=forbid_generation
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  if self.path!='/v1/systemone':self.send_error(404);return
  try:
   body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   if body.get('model','default') not in ['default','laya','jev-latest','qwen35-08b-decision','reflex-1']:raise ValueError('Unknown model alias')
   start=time.perf_counter()
   answers=FastDecisionModel.predict(model,tokenizer,body['state'],body['questions'])
   data=json.dumps({'model':'reflex-1','answers':answers,'usage':{'output_tokens':0},'latency_seconds':time.perf_counter()-start}).encode()
   self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  except (KeyError,ValueError,TypeError) as e:self.send_error(400,str(e))
print('Ready: http://127.0.0.1:'+str(a.port)+'/v1/systemone',flush=True)
HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

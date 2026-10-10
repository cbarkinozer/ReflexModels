"""Export the archived merged decision model; no training or W&B run."""
import os,json,hashlib
from pathlib import Path
os.environ.update(UNSLOTH_DISABLE_STATISTICS='1',UNSLOTH_COMPILE_DISABLE='1',WANDB_MODE='disabled',HF_HOME='/workspace/reflex-decision/cache/huggingface',UNSLOTH_LLAMA_CPP_PATH='/workspace/reflex-decision/llama.cpp',UNSLOTH_LLAMA_CPP_SCRIPTS_DIR='/workspace/reflex-decision/llama.cpp')
from unsloth.models.decision_gguf import export_decision_gguf
root=Path('/workspace/reflex-decision')
out=root/'runs/reflex-1-gguf';out.mkdir(exist_ok=True)
result=export_decision_gguf(root/'runs/notebook-60steps/merged',quantization_method=['q8_0','q4_k_m'],output_dir=out,print_output=True)
files=[]
for p in sorted(out.glob('*')):
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(out/'inventory.json').write_text(json.dumps({'release':'reflex-1','export':result,'files':files},indent=2))
print('EXPORT COMPLETE',flush=True)

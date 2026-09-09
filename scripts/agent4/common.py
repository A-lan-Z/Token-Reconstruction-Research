from pathlib import Path
import hashlib,json,subprocess,time,os
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from safetensors.torch import load_file
from token_reconstruction.public_prefix import ContiguousPublicPrefix
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/agent4-prefix-only-inversion'
EVID=ROOT/'experiments/agent4-prefix-only-inversion/evidence'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2)

def guard():
    free,total=torch.cuda.mem_get_info()
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    temperature=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    if free<2*2**30 or available<8*2**30 or torch.cuda.memory_reserved()>6*2**30 or temperature>=80:
        raise RuntimeError(f'resource guard: free={free} host={available} temp={temperature}')

def load_prefix(dtype=torch.float32,device='cuda',asset_root=None):
    assets=Path(asset_root or OUT/'backup')
    config=LlamaConfig.from_json_file(str(assets/'config.json'))
    config._attn_implementation='sdpa'
    with torch.device('meta'): full=LlamaForCausalLM(config)
    prefix=ContiguousPublicPrefix(full,4)
    del full
    prefix.rotary_emb=LlamaRotaryEmbedding(config=config,device='cpu')
    prefix.load_state_dict(load_file(str(assets/'prefix.safetensors')),strict=True,assign=True)
    prefix.to(device=device,dtype=dtype)
    # The native public loader retains nonpersistent RoPE frequencies in FP32.
    # Casting the whole module to BF16 must not quantize these public constants.
    prefix.rotary_emb=LlamaRotaryEmbedding(config=config,device=device)
    return prefix.eval().requires_grad_(False)

def environment():
    import transformers,platform
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,temperature.gpu','--format=csv'],text=True),
        'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}

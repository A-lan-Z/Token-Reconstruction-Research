from pathlib import Path
import sys,json,time,hashlib,subprocess,os
ROOT=Path(__file__).resolve().parents[2]
SIB=ROOT.parent
ASSETS=SIB/'agent4-prefix-only-inversion/outputs/agent4-prefix-only-inversion'
RESCUE=SIB/'agent4-prefix-only-rescue/outputs/agent4-prefix-only-rescue'
sys.path.insert(0,str(ROOT/'scripts/agent4'))
import common as inherited
# Loading only supplied public prefix assets; no live target or fitted decoder.
inherited.OUT=ASSETS
import torch
from safetensors.torch import load_file,save_file
from token_reconstruction.a1a2_configuration_search import _candidate_hidden
from token_reconstruction.public_prefix import ContiguousPublicPrefix
X=ROOT/'experiments/TRR-0014'
OUT=ROOT/'outputs/TRR-0014'
def load_prefix():return inherited.load_prefix(torch.bfloat16,asset_root=ASSETS/'backup')
def sync():torch.cuda.synchronize()
def digest(p):return inherited.digest(p)
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2)
def environment():return inherited.environment()
def guard():return inherited.guard()
@torch.no_grad()
def candidates(prefix,cache,ids,pos):
    return _candidate_hidden(prefix,cache=cache,parent_indices=torch.tensor([0],device='cuda'),candidate_ids=ids.reshape(1,-1),position=pos)[0]
def new_context(prefix):
    cache=prefix.new_cache()
    prefix.run_cached(torch.tensor([[128000]],device='cuda'),cache,0)
    return cache

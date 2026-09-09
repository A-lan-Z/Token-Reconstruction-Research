"""Rescue paths, immutable evidence and shared frozen public assets."""
import sys,json,time,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/agent4'))
from common import load_prefix,guard,environment,digest
import torch
X=ROOT/'experiments/agent4-prefix-only-rescue'
E=X/'evidence'
OUT=ROOT/'outputs/agent4-prefix-only-rescue'
OLD=ROOT/'outputs/agent4-prefix-only-inversion'
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2)
def sync():
    torch.cuda.synchronize()

"""Verify unchanged development sources without touching the baseline process."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage50"))
import grid50_support as dev
if dev.torch.cuda.is_initialized():raise RuntimeError("binding probe initialized CUDA")
result=dev.binding()
if dev.torch.cuda.is_initialized():raise RuntimeError("binding probe initialized CUDA")
print(json.dumps(result,sort_keys=True))

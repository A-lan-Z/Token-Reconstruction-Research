from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5","scripts/trr0020_stage6r1"]:
    sys.path.insert(0,str(ROOT/path))
from query_soft import reference_tests
import torch
torch.set_num_threads(2)
result=reference_tests()
p=ROOT/"experiments/TRR-0020/dev12_cpu_reference.json"
with p.open("x") as f:json.dump({"scope":"CPU mathematical contract only; no reconstruction performance claim","torch":torch.__version__,"time_unix":time.time(),**result},f,indent=2)
print(json.dumps(result,indent=2))

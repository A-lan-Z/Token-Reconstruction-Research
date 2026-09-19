from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5","scripts/trr0020_stage6r1","scripts/trr0020_stage7"]:
    sys.path.insert(0,str(ROOT/path))
from epsilon_soft import reference_tests
import torch
torch.set_num_threads(2)
with (ROOT/"experiments/TRR-0020/dev14_cpu_reference.json").open("x") as f:
    json.dump({"torch":torch.__version__,"time_unix":time.time(),**reference_tests()},f,indent=2)
print("CPU Adam-floor and loss-scaling identities passed")

from pathlib import Path
import sys,json,time,subprocess,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage38_grid"))
import support as inherited
for name in ["trr0020_stage44","trr0020_stage45","trr0020_stage46","trr0020_stage49_r1"]:
    sys.path.insert(0,str(ROOT/"scripts"/name))
sys.path.insert(0,str(Path(__file__).parent))
from moment_optimizer import MomentOptimizer
from first_optimizer import gradient_reference
from pointwise import bits_equal
n=inherited.n;torch=n.torch;X=inherited.X;INPUT=inherited.INPUT;guard=inherited.guard;rows=inherited.rows
OUT=ROOT/"outputs/TRR-0020/dev50"
MODES=["two_point","bennett","control"];STEPS=[128,64,32,16]
CONFIGS=[(f"{mode}_steps{steps}",mode,steps) for mode in MODES for steps in STEPS]
AUX=inherited.AUX
def binding():
    paths=list(Path(__file__).parent.glob("*.py"))
    for name in ["trr0020_stage44","trr0020_stage45","trr0020_stage46","trr0020_stage49_r1"]:
        paths+=list((ROOT/"scripts"/name).glob("*.py"))
    paths +=[X/name for name in ["DEV50_PLAN.md","dev50_preflight.json","dev49_r1_cpu_reference.json","dev49_r1_gpu_check.json","dev48_r1_freeze.json","dev48_r1_phase_control.json","dev46_freeze.json","dev46_phase_reuse.json"]]
    return {"inherited":inherited.binding(),"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths}}
def trace_equal(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def equal(a,sa,b,sb):
    return set(a)==set(b) and all(bits_equal(v,b[k]) for k,v in a.items()) and all(trace_equal(sa["warm"][k],sb["warm"][k]) for k in inherited.old.TRACE_KEYS) and trace_equal(sa["mirror_traces"],sb["mirror_traces"])
def anchor(a,sa,b,sb,steps,control):
    if control:return equal(a,sa,b,sb)
    return all(bits_equal(v,b[k]) for k,v in a.items() if k.startswith("warm_") or k=="initial_embedding") and all(trace_equal(sa["warm"][k],sb["warm"][k]) for k in inherited.old.TRACE_KEYS)
def verify(e,bound):
    if e["binding"]!=bound:raise ValueError("binding")
    for rep in e["repetitions"]:
        if not rep["exact_repeat"] or not rep["anchor_valid"] or n.digest(ROOT/rep["path"])!=rep["sha256"]:raise ValueError("changed replicate")
    if n.digest(ROOT/e["path"])!=e["sha256"]:raise ValueError("output hash")
    return n.load_file(str(ROOT/e["path"]))

from pathlib import Path
import sys,json,time,subprocess,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage38_grid"))
import support as inherited
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage44"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage45"))
sys.path.insert(0,str(Path(__file__).parent))
from reused_optimizer import ReusedOptimizer
from first_optimizer import gradient_reference
n=inherited.n;torch=n.torch;X=inherited.X;INPUT=inherited.INPUT;guard=inherited.guard;rows=inherited.rows
OUT=ROOT/"outputs/TRR-0020/dev46"
STEPS=[128,64]
CONFIGS=[(f"{mode}_steps{steps}",mode,steps) for mode in ["control","reuse"] for steps in STEPS]
AUX=inherited.AUX
def binding():
    paths=list(Path(__file__).parent.glob("*.py"))+list((ROOT/"scripts/trr0020_stage44").glob("*.py"))+list((ROOT/"scripts/trr0020_stage45").glob("*.py"))
    paths +=[X/v for v in ["DEV46_PLAN.md","dev46_preflight.json","dev46_cpu_reference.json","dev46_gpu_check.json","dev45_freeze.json","dev45_phase_exact.json"]]
    return {"inherited":inherited.binding(),"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths}}
from pointwise import bits_equal
def trace_equal(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def equal(a,sa,b,sb):
    return (set(a)==set(b) and all(bits_equal(v,b[k]) for k,v in a.items()) and
      all(trace_equal(sa["warm"][k],sb["warm"][k]) for k in inherited.old.TRACE_KEYS) and
      trace_equal(sa["mirror_traces"],sb["mirror_traces"]))
def anchor(output,stats,ref,rs,steps,control):
    # Both execution variants must reproduce the complete archived original output.
    return equal(output,stats,ref,rs)
def verify(e,bound):
    if e["binding"]!=bound:raise ValueError("binding")
    for rep in e["repetitions"]:
        if not rep["exact_repeat"] or not rep["anchor_valid"] or n.digest(ROOT/rep["path"])!=rep["sha256"]:raise ValueError("changed replicate")
    if n.digest(ROOT/e["path"])!=e["sha256"]:raise ValueError("output hash")
    return n.load_file(str(ROOT/e["path"]))

from pathlib import Path
import sys,json,time,subprocess,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage38_grid"))
import support as inherited
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage42"))
sys.path.insert(0,str(Path(__file__).parent))
from fused_optimizer import FusedOptimizer
from first_optimizer import gradient_reference
n=inherited.n;torch=n.torch;X=inherited.X;INPUT=inherited.INPUT;guard=inherited.guard;rows=inherited.rows
OUT=ROOT/"outputs/TRR-0020/dev43"
STEPS=[128,64]
CONFIGS=[(f"{mode}_steps{steps}",mode,steps) for mode in ["control","fused"] for steps in STEPS]
AUX=inherited.AUX
def binding():
    paths=list(Path(__file__).parent.glob("*.py"))+list((ROOT/"scripts/trr0020_stage42").glob("*.py"))
    paths +=[X/v for v in ["DEV43_PLAN.md","dev43_preflight.json","dev42_cpu_reference.json","dev42_gpu_check.json","dev42_gpu_summary.json","dev42_gpu_archive.json","dev38_grid_freeze.json"]]
    return {"inherited":inherited.binding(),"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths}}
def equal(a,sa,b,sb):return inherited.equal(a,sa,b,sb)
def anchor(output,stats,ref,rs,steps,control):
    good=(all(torch.equal(v,ref[k]) for k,v in output.items() if k.startswith("warm_")) and
      all(stats["warm"][k]==rs["warm"][k] for k in inherited.old.TRACE_KEYS) and
      torch.equal(output["initial_embedding"],ref["initial_embedding"]))
    if control:
        good=good and torch.equal(output["step64"],ref["step64"])
        good=good and all(stats["mirror_traces"][k][:65]==rs["mirror_traces"][k] for k in stats["mirror_traces"])
        good=good and torch.equal(output["mirror_update_trace"][:64],ref["mirror_update_trace"])
        if steps==64:good=good and set(output)==set(ref) and all(torch.equal(v,ref[k]) for k,v in output.items())
    return good
def verify(e,bound):
    if e["binding"]!=bound:raise ValueError("binding")
    for rep in e["repetitions"]:
        if not rep["exact_repeat"] or not rep["anchor_valid"] or n.digest(ROOT/rep["path"])!=rep["sha256"]:raise ValueError("changed replicate")
    if n.digest(ROOT/e["path"])!=e["sha256"]:raise ValueError("output hash")
    return n.load_file(str(ROOT/e["path"]))

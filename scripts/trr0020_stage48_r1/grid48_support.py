from pathlib import Path
import sys,json,time,subprocess,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage38_grid"))
import support as inherited
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage44"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage45"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage46"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage47_r1"))
sys.path.insert(0,str(Path(__file__).parent))
from history_optimizer import HistoryOptimizer,CONFIGS as HISTORY_CONFIGS
from first_optimizer import gradient_reference
n=inherited.n;torch=n.torch;X=inherited.X;INPUT=inherited.INPUT;guard=inherited.guard;rows=inherited.rows
OUT=ROOT/"outputs/TRR-0020/dev48_r1"
STEPS=[64,32,16]
CONFIGS=[(f"{mode}_steps{steps}",mode,steps) for mode in HISTORY_CONFIGS for steps in STEPS]
AUX=inherited.AUX|{"aa_trace"}
def binding():
    paths=list(Path(__file__).parent.glob("*.py"))+list((ROOT/"scripts/trr0020_stage44").glob("*.py"))+list((ROOT/"scripts/trr0020_stage45").glob("*.py"))+list((ROOT/"scripts/trr0020_stage46").glob("*.py"))+list((ROOT/"scripts/trr0020_stage47_r1").glob("*.py"))
    paths +=[X/v for v in ["DEV48_R1_PLAN.md","dev48_r1_preflight.json","dev48_r1_cpu_reference.json","dev47_r1_public_freeze.json","dev47_r1_public_score.json","dev46_freeze.json","dev48_freeze.json","dev48_numerics_audit.json"]]
    return {"inherited":inherited.binding(),"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths}}
from pointwise import bits_equal
def trace_equal(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def equal(a,sa,b,sb):
    return (set(a)==set(b) and all(bits_equal(v,b[k]) for k,v in a.items()) and
      all(trace_equal(sa["warm"][k],sb["warm"][k]) for k in inherited.old.TRACE_KEYS) and
      trace_equal(sa["mirror_traces"],sb["mirror_traces"]))
def warm(output,stats,ref,rs):
    return (all(bits_equal(v,ref[k]) for k,v in output.items() if k.startswith("warm_") or k=="initial_embedding") and
      all(trace_equal(stats["warm"][k],rs["warm"][k]) for k in inherited.old.TRACE_KEYS))
def compatible_equal(output,stats,ref,rs):
    a={k:v for k,v in output.items() if k!="aa_trace"};b={k:v for k,v in ref.items() if k!="aa_trace"}
    good=equal(a,stats,b,rs)
    if "aa_trace" in output:
        good=good and bits_equal(output["aa_trace"][:,:3],ref["aa_trace"])
        good=good and bool((output["aa_trace"][:,3:]==1).all())
    return good
def anchor(output,stats,ref,rs,steps,control):
    if stats["history_rule"]=="aa_logit_clip1_scaled":return warm(output,stats,ref,rs)
    good=warm(output,stats,ref,rs)
    good=good and bits_equal(output["step"+str(steps)],ref["step"+str(steps)])
    good=good and bits_equal(output["mirror_update_trace"],ref["mirror_update_trace"][:steps])
    good=good and all(trace_equal(stats["mirror_traces"][k],v[:steps+1]) for k,v in rs["mirror_traces"].items())
    if "aa_trace" in output:
        good=good and bits_equal(output["aa_trace"][:,:3],ref["aa_trace"][:steps]) and bool((output["aa_trace"][:,3:]==1).all())
    return good and (steps!=64 or compatible_equal(output,stats,ref,rs))
def real_anchor(output,stats,ref,rs,steps,control):
    if stats["history_rule"]=="aa_logit_clip1_scaled":return warm(output,stats,ref,rs)
    return compatible_equal(output,stats,ref,rs)
def verify(e,bound):
    if e["binding"]!=bound:raise ValueError("binding")
    for rep in e["repetitions"]:
        if not rep["exact_repeat"] or not rep["anchor_valid"] or n.digest(ROOT/rep["path"])!=rep["sha256"]:raise ValueError("changed replicate")
    if n.digest(ROOT/e["path"])!=e["sha256"]:raise ValueError("output hash")
    return n.load_file(str(ROOT/e["path"]))

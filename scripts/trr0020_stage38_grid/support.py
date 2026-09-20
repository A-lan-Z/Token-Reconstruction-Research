from pathlib import Path
import sys,json,time,subprocess,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage32"))
import grid_support as old
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage38"))
from first_optimizer import FirstOptimizer,gradient_reference
n=old.n;torch=old.torch;X=old.X;INPUT=old.INPUT;guard=old.guard;rows=old.rows
OUT=ROOT/"outputs/TRR-0020/dev38_grid"
STEPS=[64,32,16]
CONFIGS=[(f"{mode}_steps{steps}",mode,steps) for mode in ["control","locked"] for steps in STEPS]
AUX={"initial_embedding","base_initial_embedding","first_choice","mirror_update_trace","final_confidence","final_position_error"}
def binding():
    inherited=old.binding()
    paths=list(Path(__file__).parent.glob("*.py"))+[ROOT/"scripts/trr0020_stage38/lookup.py"]
    paths +=[X/v for v in ["DEV38_GRID_PLAN.md","dev38_grid_preflight.json","dev38_public_probe.json",
      "dev38_public_summary.json","dev38_public_archive.json","dev32_grid_freeze.json"]]
    source=json.loads((X/"dev38_public_probe.json").read_text())["table_source"]
    return {"inherited":inherited,"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "table_sha256":source["sha256"],"table_path":source["path"]}
def equal(a,sa,b,sb):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a) and all(sa["warm"][k]==sb["warm"][k] for k in old.TRACE_KEYS) and sa["mirror_traces"]==sb["mirror_traces"]
def warm_anchor(output,stats,reference,rs,steps,locked):
    exact=all(torch.equal(v,reference[k]) for k,v in output.items() if k.startswith("warm_"))
    exact=exact and all(stats["warm"][k]==rs["warm"][k] for k in old.TRACE_KEYS)
    raw=output["base_initial_embedding"] if locked else output["initial_embedding"]
    exact=exact and torch.equal(raw,reference["initial_embedding"])
    if not locked:
        exact=exact and all(torch.equal(v,reference[k]) for k,v in output.items()
          if k.startswith("step")) and all(stats["mirror_traces"][k]==rs["mirror_traces"][k][:steps+1] for k in stats["mirror_traces"])
        if steps==64:exact=exact and set(output)==set(reference) and all(torch.equal(v,reference[k]) for k,v in output.items())
    return exact
def check_fixed(output,engine,locked):
    if not locked:return True
    first=output["first_choice"]
    keys=[k for k in output if k.startswith("step") or k in ["best_objective","best_position_error"]]
    return (all(torch.equal(output[k][1:2],first) for k in keys)
      and torch.equal(output["initial_embedding"][:1],engine.weight.index_select(0,first.to("cuda")).float().cpu()))
def verify(entry,bound):
    if entry["binding"]!=bound:raise ValueError("binding changed")
    for rep in entry["repetitions"]:
        if n.digest(ROOT/rep["path"])!=rep["sha256"] or not rep["exact_repeat"]:raise ValueError("changed replicate")
    if n.digest(ROOT/entry["path"])!=entry["sha256"]:raise ValueError("changed output")
    return n.load_file(str(ROOT/entry["path"]))

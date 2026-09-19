"""Summarize the scored fixed refinement comparison without pooling setups."""
from pathlib import Path
import json,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1]
X=ROOT/"experiments/TRR-0020/canonical_refine"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    score=json.loads((X/"score.json").read_text())
    matrix=json.loads((X/"canonical_matrix.json").read_text())
    guard=json.loads((X/"guard.json").read_text())
    if not matrix["complete"] or matrix["required_cells"]!=68:raise RuntimeError("incomplete matrix")
    if guard["returncode"] or guard["failure"] is not None:raise RuntimeError("guard failure")
    phases=[json.loads(p.read_text()) for p in sorted(X.glob("phase_*.json"))]
    if len(phases)!=9:raise RuntimeError("expected nine phases")
    resources={}
    for method in ["a1_native","a1_graph","warm128white003"]:
        ps=[p for p in phases if p["method"]==method]
        resources[method]={"peak_reserved_GiB":max(p["peak_reserved"] for p in ps)/2**30,
          "peak_allocated_GiB":max(p["peak_allocated"] for p in ps)/2**30,
          "preparation":[p["setup"] for p in ps],"execution_commits":sorted({p["code_commit"] for p in ps})}
    native=[p for p in score["paired"] if p["control"]=="a1_native"]
    result={"summary":score["summary"],"paired":score["paired"],"resources":resources,
      "token_score_not_lower_than_native_in_both":all(p["correct_delta"]>=0 for p in native),
      "exact_inputs_not_lower_than_native_in_both":all(p["exact_delta"]>=0 for p in native),
      "inference_not_slower_than_native_in_both":all(p["runtime_ratio"]<=1 for p in native),
      "guard_wall_seconds":guard["end_unix"]-guard["start_unix"],
      "min_gpu_free_MiB":min(s["gpu_free_mib"] for s in guard["samples"]),
      "max_temperature_C":max(s["temperature_c"] for s in guard["samples"]),
      "source_score_sha256":digest(X/"score.json"),"source_matrix_sha256":digest(X/"canonical_matrix.json"),
      "scope":"Retrospective supplied-prefix comparison; no pooled benchmark score or automatic goal-completion decision"}
    with (X/"summary.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({"summary":result["summary"],"paired":result["paired"],
      "resources":{k:v["peak_reserved_GiB"] for k,v in resources.items()},
      "guard_wall_seconds":result["guard_wall_seconds"]},indent=2))
if __name__=="__main__":main()

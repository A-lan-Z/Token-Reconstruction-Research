"""Descriptive paired public-prefix comparison; no token-accuracy claims."""
from pathlib import Path
import json,statistics,time,hashlib
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    q=json.loads((X/"dev34_public_probe.json").read_text());g=json.loads((X/"dev34_guard.json").read_text())
    a=json.loads((X/"dev34_public_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or g["failure"] is not None:raise RuntimeError("incomplete public run")
    rows=[]
    for cell in q["cells"]:
        reps=cell["repetitions"];first=reps[0]
        if len(reps)!=3 or not all(v["passed"] and v["repeat_equal"] for v in reps):raise RuntimeError("invalid repeats")
        row={k:cell[k] for k in ["family","length","position","method","rule","steps"]}
        row.update(observed_error=first["observed_error"],relative_error=first["relative_error"],confidence=first["confidence"],
          prefix_forwards=first["prefix_forwards"],prefix_vjps=first["prefix_vjps"],
          median_inference_seconds=statistics.median(v["inference_seconds"] for v in reps),
          median_commit_seconds=statistics.median(v["emitted_token_commit_seconds"] for v in reps))
        if cell["rule"]=="backtrack":row.update(accepted_steps=first["accepted_steps"],rejected_trials=first["rejected_trials"])
        rows.append(row)
    comparisons=[]
    for context in q["contexts"]:
      for steps in [8,32]:
        select=lambda rule:next(v for v in rows if all(v[k]==context[k] for k in ["family","length","position"]) and v["steps"]==steps and v["rule"]==rule)
        fixed=select("fixed");new=select("backtrack")
        comparisons.append({k:context[k] for k in ["family","length","position"]}|{"steps":steps,"initial_error":context["initial_error"],
          "fixed_error":fixed["observed_error"],"backtrack_error":new["observed_error"],"error_difference":new["observed_error"]-fixed["observed_error"],
          "fixed_forwards":fixed["prefix_forwards"],"backtrack_forwards":new["prefix_forwards"],"backtrack_rejections":new["rejected_trials"],
          "fixed_seconds":fixed["median_inference_seconds"],"backtrack_seconds":new["median_inference_seconds"]})
    result={"task_id":"TRR-0020","scope":"public synthetic fixed-history activation matching; no reconstruction accuracy",
      "cells":len(rows),"repetitions":sum(len(c["repetitions"]) for c in q["cells"]),"independent_gradients":len(q["gradient_references"]),
      "maximum_gradient_error":max(v["gradient_max_error"] for v in q["gradient_references"]),
      "old_fixed8_anchors":len(q["old_fixed8_anchors"]),"rows":rows,"comparisons":comparisons,
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "guard_returncode":g["returncode"],"archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],
      "archive_bytes":a["raw_archive_bytes"],"probe_sha256":sha(X/"dev34_public_probe.json"),"created_unix":time.time()}
    with (X/"dev34_public_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    for v in comparisons:print(json.dumps(v))
    print(json.dumps({k:v for k,v in result.items() if k not in ["rows","comparisons"]},indent=2))
if __name__=="__main__":main()

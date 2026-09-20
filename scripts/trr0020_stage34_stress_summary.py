"""Summarize public common-token sanity checks, separate from benchmark accuracy."""
from pathlib import Path
import json,time,statistics
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev34_stress_probe.json").read_text());g=json.loads((X/"dev34_stress_guard.json").read_text())
    archive=json.loads((X/"dev34_stress_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or g["failure"] is not None:raise RuntimeError("incomplete qualification")
    rows=[];rejected_contexts=set()
    for c in q["cells"]:
        context=next(v for v in q["contexts"] if v["precision"]==c["precision"] and v["context"]==c["context"])
        reps=c["repetitions"];first=reps[0]
        with safe_open(ROOT/first["path"],framework="pt",device="cpu") as f:
            if c["rule"]=="fixed":
                trace=f.get_tensor("trace")[:,0].tolist()+f.get_tensor("observed_error").tolist()
                increases=sum(b>a+1e-7 for a,b in zip(trace,trace[1:]))
            else:
                trace=f.get_tensor("step_trace");increases=int((trace[:,1]>trace[:,0]+1e-7).sum())
        row={k:c[k] for k in ["precision","context","method","rule","steps"]}
        row.update(public_text=context["public_text"],identified=first["emitted_token_id"]==context["public_token_id"],
          initial_error=context["initial_error"],observed_error=first["observed_error"],confidence=first["confidence"],
          prefix_forwards=first["prefix_forwards"],prefix_vjps=first["prefix_vjps"],loss_increases=increases,
          rejected_trials=first.get("rejected_trials",0),median_seconds=statistics.median(v["inference_seconds"] for v in reps))
        if row["rejected_trials"]:rejected_contexts.add((row["precision"],row["context"]))
        rows.append(row)
    comparisons=[]
    for context in q["contexts"]:
      for steps in [8,32]:
        select=lambda rule:next(v for v in rows if v["precision"]==context["precision"] and v["context"]==context["context"] and v["steps"]==steps and v["rule"]==rule)
        fixed,new=select("fixed"),select("backtrack")
        comparisons.append({"precision":context["precision"],"context":context["context"],"public_text":context["public_text"],"steps":steps,
          "fixed_error":fixed["observed_error"],"backtrack_error":new["observed_error"],"fixed_identified":fixed["identified"],"backtrack_identified":new["identified"],
          "fixed_forwards":fixed["prefix_forwards"],"backtrack_forwards":new["prefix_forwards"],"fixed_loss_increases":fixed["loss_increases"],
          "backtrack_rejections":new["rejected_trials"],"fixed_seconds":fixed["median_seconds"],"backtrack_seconds":new["median_seconds"]})
    result={"task_id":"TRR-0020","scope":"public common-token numerical sanity; not benchmark reconstruction accuracy",
      "cells":len(q["cells"]),"repetitions":sum(len(c["repetitions"]) for c in q["cells"]),
      "gradient_references":len(q["gradient_references"]),"gradient_max_error":max(v["gradient_max_error"] for v in q["gradient_references"]),
      "public_identity_sanity":q["public_identity_sanity"],"contexts_with_rejections":len(rejected_contexts),
      "rows":rows,"comparisons":comparisons,"peak_reserved_GiB":q["peak_reserved"]/2**30,
      "guard_returncode":g["returncode"],"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "archive_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],"archive_bytes":archive["raw_archive_bytes"],
      "created_unix":time.time()}
    with (X/"dev34_stress_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    for v in comparisons:
        if v["backtrack_rejections"] or not v["fixed_identified"] or not v["backtrack_identified"]:print(json.dumps(v))
    print(json.dumps({k:v for k,v in result.items() if k not in ["rows","comparisons"]},indent=2))
if __name__=="__main__":main()

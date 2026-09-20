"""Summarize the complete public KL diagnostic without reconstruction claims."""
from pathlib import Path
import json,statistics,time,hashlib
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
 q=json.loads((X/"dev31_public_probe_r1.json").read_text())
 guard=json.loads((X/"dev31_guard_r1.json").read_text())
 archive=json.loads((X/"dev31_public_archive.json").read_text())
 if not q["passed"] or guard["returncode"] or guard["failure"]:raise RuntimeError("incomplete run")
 rows=[]
 for context in q["contexts"]:
  for tau in [.01,.1,1.]:
   row={"length":context["length"],"warm":context["warm"],"tau":tau,"initial_error":context["initial_observed_error"]}
   for name in ["span4","forward_kl"]:
    cell=next(c for c in q["cells"] if c["length"]==context["length"] and c["warm"]==context["warm"] and c["tau"]==tau and c["rule"]==name)
    v=cell["repetitions"][0]
    row[name]={key:v[key] for key in ["observed_error","relative_observed_error","mean_confidence","kl_min","kl_mean","kl_max","positions_error_improved"]}
    row[name]["median_update_seconds"]=statistics.median(r["update_seconds"] for r in cell["repetitions"])
   rows.append(row)
 result={"task_id":"TRR-0020","scope":"public synthetic one-step diagnostic; no reconstruction accuracy or end-to-end decoder claim",
  "cases":len(q["cells"]),"repetitions":sum(len(c["repetitions"]) for c in q["cells"]),"warm_anchors":len(q["contexts"]),
  "gradient_references":len(q["contexts"]),"gradient_max_absolute_error":max(c["gradient_reference"]["max_absolute_error"] for c in q["contexts"]),
  "original_update_checks":sum(len(c["repetitions"]) for c in q["cells"] if c["rule"]=="span4"),
  "repair_saved_artifact_checks":q["completed_attempt_anchor_checks"],"rows":rows,
  "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_returncode":guard["returncode"],"guard_seconds":guard["end_unix"]-guard["start_unix"],
  "min_free_mib":min(v["gpu_free_mib"] for v in guard["samples"]),"max_temperature_c":max(v["temperature_c"] for v in guard["samples"]),
  "archive_bytes":archive["raw_archive_bytes"],"archive_logical_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],
  "excluded_attempt":{"probe":"dev31_public_probe.json","guard":"dev31_guard.json","complete_cases":12,"reason":"retained eager autograd graph prevented next-length CUDA capture; preserved and exactly reproduced"},
  "conclusion":"Larger KL improves initial-state progress but damages already-good positions. Fixed KL cost is high; test per-position observed-error budgets and explicitly approximate shorter scalar solves.",
  "sources":{name:hashlib.sha256((X/name).read_bytes()).hexdigest() for name in ["dev31_public_probe_r1.json","dev31_guard_r1.json","dev31_public_archive.json"]},
  "created_unix":time.time()}
 p=X/"dev31_public_summary.json"
 with p.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
 print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()

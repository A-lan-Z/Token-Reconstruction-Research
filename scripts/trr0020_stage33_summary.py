"""Descriptive public current-token update summary."""
from pathlib import Path
import json,statistics,time
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
 q=json.loads((X/"dev33_public_probe.json").read_text());guard=json.loads((X/"dev33_guard.json").read_text())
 archive=json.loads((X/"dev33_public_archive.json").read_text())
 if not q["passed"] or guard["returncode"] or guard["failure"]:raise RuntimeError("incomplete public evidence")
 rows=[]
 for cell in q["cells"]:
  v=cell["repetitions"][0];rows.append({k:cell[k] for k in ["length","position","factor","steps"]}|
    {k:v[k] for k in ["observed_error","relative_error","confidence"]}|
    {"median_inference_seconds":statistics.median(r["inference_seconds"] for r in cell["repetitions"]),
     "median_commit_seconds":statistics.median(r["emitted_token_commit_seconds"] for r in cell["repetitions"])})
 phases={name:statistics.median(rep["phase_gpu_seconds"][name]/cell["steps"] for cell in q["cells"] for rep in cell["repetitions"]) for name in q["cells"][0]["repetitions"][0]["phase_gpu_seconds"]}
 result={"task_id":"TRR-0020","scope":q["scope"],"cases":len(q["cells"]),"repetitions":sum(len(c["repetitions"]) for c in q["cells"]),
  "forward_anchors":q["forward_anchors"],"derivative_anchors":q["derivative_anchors"],
  "independent_probability_gradients":len(q["probability_gradient_references"]),
  "maximum_probability_gradient_error":max(c["gradient_max_absolute_error"] for c in q["probability_gradient_references"]),
  "rows":rows,"median_component_seconds_per_update":phases,"peak_reserved_GiB":q["peak_reserved"]/2**30,
  "guard_returncode":guard["returncode"],"guard_seconds":guard["end_unix"]-guard["start_unix"],
  "minimum_free_mib":min(v["gpu_free_mib"] for v in guard["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in guard["samples"]),
  "archive_bytes":archive["raw_archive_bytes"],"archive_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],
  "conclusion":"All six fixed-history contexts converge substantially under factor2, but eager timing is costly. Qualify exact replay and full native128-position geometry before reconstruction claims.",
  "created_unix":time.time()}
 p=X/"dev33_public_summary.json"
 with p.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
 print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()

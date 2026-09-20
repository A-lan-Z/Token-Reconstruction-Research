"""Summarize qualified whole-vocabulary scalar fusion and isolated timing."""
from pathlib import Path
import json,statistics,time,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev42_gpu_check.json").read_text());g=json.loads((X/"dev42_gpu_guard.json").read_text());a=json.loads((X/"dev42_gpu_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or not a["all_members_roundtrip_verified"]:raise RuntimeError("incomplete")
    timings=[]
    for b in q["benchmarks"]:
        old=statistics.median(v["seconds_per_call"] for v in b["groups"] if v["method"]=="original")
        new=statistics.median(v["seconds_per_call"] for v in b["groups"] if v["method"]=="fused")
        timings.append({"label":b["label"],"original_seconds":old,"fused_seconds":new,"ratio":new/old,"speedup":old/new,"setup":b["setup"]})
    reps=[rep for cell in q["cells"] for rep in cell["repetitions"]]
    checks=[v for cell in q["cells"] for v in cell["scalar_checks"]]
    result={"task_id":"TRR-0020","scope":q["scope"],"cells":len(q["cells"]),"scalar_checks":len(checks),"full_update_repetitions":len(reps),"timed_graph_calls":240,
      "maximum_scalar_value_error":max(v["maximum_value_error"] for v in checks),
      "maximum_scalar_derivative_error":max(v["maximum_derivative_error"] for v in checks),
      "maximum_probability_TV":max(v["probability_TV"] for v in reps),"maximum_logit_error":max(v["maximum_logit_error"] for v in reps),
      "maximum_step_error":max(v["maximum_step_error"] for v in reps),"maximum_KL_minus_budget":max(v["max_KL_minus_budget"] for v in reps),
      "all_repetitions_exact":all(v["exact_repeat"] for v in reps),"all_original_anchors_exact":all(v["original_anchor_exact"] and v["refactoring_exact"] for v in q["cells"]),
      "timings":timings,"peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],"archive_bytes":a["raw_archive_bytes"],
      "limitation":"Isolated warmed scalar-update timing only; full decoder speed and accuracy remain unmeasured for this numerical variant.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev42_gpu_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

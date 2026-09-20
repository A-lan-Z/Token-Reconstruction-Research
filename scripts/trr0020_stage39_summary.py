"""Summarize public derivatives and linear systems; no nonlinear decoder claim."""
from pathlib import Path
import json,time,statistics,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev39_gpu_qualification.json").read_text());g=json.loads((X/"dev39_gpu_guard.json").read_text())
    a=json.loads((X/"dev39_gpu_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or g["failure"] is not None:raise RuntimeError("incomplete")
    rows=[]
    for v in q["linear_cells"]:
        reps=v["repetitions"]
        if len(reps)!=3 or not all(r["passed"] and r["exact_repeat"] for r in reps):raise RuntimeError("invalid repeats")
        rows.append({k:v[k] for k in ["length","point","strength","steps"]}|
          {"median_seconds":statistics.median(r["seconds"] for r in reps),"relative_residual":reps[0]["relative_residual"],
           "direction_norm":reps[0]["direction_norm"],"jvp_calls":reps[0]["matvec_calls"]})
    out={"task_id":"TRR-0020","scope":q["scope"],"derivative_cells":len(q["derivative_cells"]),"derivative_checks":3*len(q["derivative_cells"]),
      "linear_cells":len(rows),"linear_repetitions":3*len(rows),"linear_rows":rows,
      "maximum_forward_error":max(v["forward_maximum_error"] for v in q["derivative_cells"]),
      "derivative_maxima":{k:max(v[k] for v in q["derivative_cells"]) for k in
        ["input_maximum_error","input_relative_norm_error","strength_maximum_error","strength_relative_norm_error","mixed_maximum_error","mixed_relative_norm_error"]},
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "preparation_seconds":q["preparation_seconds"],"archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],
      "archive_bytes":a["raw_archive_bytes"],"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev39_gpu_summary.json").open("x") as f:json.dump(out,f,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k!="linear_rows"},indent=2))
    for row in rows:
        if row["steps"]==16 or (row["strength"]==1. and row["steps"]==4):print(json.dumps(row))
if __name__=="__main__":main()

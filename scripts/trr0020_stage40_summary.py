"""Summarize the complete public Krylov audit without a benchmark claim."""
from pathlib import Path
import json,statistics,time,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev40_gpu_audit.json").read_text());a=json.loads((X/"dev40_gpu_archive.json").read_text());g=json.loads((X/"dev40_gpu_guard.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or not a["all_members_roundtrip_verified"]:raise RuntimeError("incomplete")
    rows=[]
    for point in q["points"]:
      for steps in [16,32]:
       for mode in ["normal_ridge","svd_ridge","svd_unregularized"]:
        comparisons=[next(v for v in rep["comparisons"] if v["steps"]==steps and v["mode"]==mode) for rep in point["repetitions"]]
        values=comparisons[0]
        rows.append({k:point[k] for k in ["length","point","strength"]}|{k:v for k,v in values.items() if not k.endswith("_seconds")}|
          {"median_32_arnoldi_seconds":statistics.median(rep["arnoldi_seconds"] for rep in point["repetitions"]),
           "median_reduced_solve_seconds":statistics.median(v["solve_seconds"] for v in comparisons),
           "median_residual_check_seconds":statistics.median(v["residual_check_seconds"] for v in comparisons)})
    result={"task_id":"TRR-0020","scope":q["scope"],"points":len(q["points"]),"solver_cells":len(rows),"solver_repetitions":len(rows)*3,
      "all_original_and_prefix_anchors_exact":all(v["original_exact"] and v["prefix_exact"] for v in q["points"]),
      "all_repeats_exact":all(rep["exact_repeat"] for v in q["points"] for rep in v["repetitions"]),
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(s["gpu_free_mib"] for s in g["samples"]),"maximum_temperature_c":max(s["temperature_c"] for s in g["samples"]),
      "maximum_basis_orthogonality_error":max(rep["basis_orthogonality_maximum_error"] for v in q["points"] for rep in v["repetitions"]),
      "preparation_seconds":q["preparation_seconds"],"archive_bytes":a["raw_archive_bytes"],"archive_members":a["logical_member_count"],
      "archive_unique_objects":a["unique_objects"],"rows":rows,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev40_gpu_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
    for point in q["points"]:
      selected=[v for v in rows if all(v[k]==point[k] for k in ["length","point","strength"])]
      print(point["length"],point["point"],point["strength"],[(v["steps"],v["mode"],round(v["relative_residual"],7),round(v["condition"],2)) for v in selected])
if __name__=="__main__":main()

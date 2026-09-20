"""Exact full-vocabulary update qualification and paired isolated timing."""
from pathlib import Path
import json,statistics,time,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev44_gpu_check.json").read_text());g=json.loads((X/"dev44_guard.json").read_text());a=json.loads((X/"dev44_gpu_archive.json").read_text())
    fresh=[json.loads((X/f"dev44_fixture_{v}.json").read_text()) for v in [128,40]]
    if not q["passed"] or g["returncode"]!=0 or not a["all_members_roundtrip_verified"] or not all(v["passed"] for v in fresh):raise RuntimeError("incomplete")
    timing=[]
    for b in q["benchmarks"]:
        old=statistics.median(v["seconds_per_call"] for v in b["groups"] if v["method"]=="original")
        new=statistics.median(v["seconds_per_call"] for v in b["groups"] if v["method"]=="exact_fused")
        timing.append({"label":b["label"],"original_seconds":old,"exact_fused_seconds":new,"ratio":new/old,"speedup":old/new,"setup":b["setup"]})
    reps=[v for c in q["cells"] for v in c["repetitions"]]
    result={"task_id":"TRR-0020","scope":q["scope"],"cases":len(q["cells"]),"primitive_checks":len(q["primitive_checks"]),
      "full_update_repetitions":len(reps),"timed_graph_calls":sum(v["calls"] for b in q["benchmarks"] for v in b["groups"]),
      "all_primitive_bytes_equal":all(v["all_bytes_equal"] for v in q["primitive_checks"]),
      "all_complete_update_bytes_equal":all(v["all_bytes_equal"] for v in reps),
      "all_graph_bytes_equal":all(v["all_bytes_equal"] for b in q["benchmarks"] for v in b["groups"]),
      "fresh_original_fixtures":sum(len(v["fixtures"]) for v in fresh),
      "all_fresh_anchors_exact":all(f["anchor_exact"] for v in fresh for f in v["fixtures"]),
      "timings":timing,"kernel_peak_reserved_GiB":q["peak_reserved"]/2**30,
      "preparation_peak_reserved_GiB":max(v["peak_reserved"] for v in fresh)/2**30,
      "guard_seconds":g["end_unix"]-g["start_unix"],"minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),
      "maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],"archive_bytes":a["raw_archive_bytes"],
      "limitation":"Exact isolated updates and public state preparation only; complete optimized decoder trajectories and reconstruction speed remain unqualified.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev44_summary.json").open("x") as z:json.dump(result,z,indent=2);z.write("\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

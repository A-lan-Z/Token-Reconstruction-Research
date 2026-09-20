"""Measured public execution cost of the unchanged full-vocabulary optimizer."""
from pathlib import Path
import json,statistics,time,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev41_profile_r2.json").read_text());g=json.loads((X/"dev41_profile_r2_guard.json").read_text())
    a=json.loads((X/"dev41_profile_r2_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or not a["all_members_roundtrip_verified"]:raise RuntimeError("incomplete")
    rows=[]
    for length in [128,40]:
        full=next(v for v in q["profiles"] if v["name"]==f"full_{length}")
        budget=next(v for v in q["profiles"] if v["name"]==f"budget_{length}")
        rows.append({"length":length,"median_public_64step_seconds":statistics.median(v["stats"]["total_seconds"] for v in q["runs"] if v["length"]==length),
          "profiled_full_update_gpu_ms":full["self_gpu_microseconds_sum"]/5000,
          "profiled_budget_update_gpu_ms":budget["self_gpu_microseconds_sum"]/5000,
          "isolated_budget_to_full_work_ratio":budget["self_gpu_microseconds_sum"]/full["self_gpu_microseconds_sum"],
          "full_top10":full["rows"][:10],"budget_top10":budget["rows"][:10]})
    result={"task_id":"TRR-0020","scope":q["scope"],"rows":rows,"anchors_and_repetitions_exact":all(v["anchor_exact"] for v in q["runs"]),
      "update_fixtures_exact_and_unchanged":all(v["repeats_exact"] and v["inputs_unchanged"] for v in q["update_fixtures"]),
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],"archive_bytes":a["raw_archive_bytes"],
      "prefix_seconds":q["prefix_seconds"],"engine_seconds":q["engine_seconds"],
      "limitation":"Profiler changes scheduling/latency; isolated budget repeatedly uses a fixed step16state while full steps evolve. These timings are diagnostics, not a speedup or canonical comparison.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev41_profile_r2_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

"""Paired isolated reuse timings, explicitly excluding full decoder conclusions."""
from pathlib import Path
import json,time,subprocess,statistics,sys
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev46_gpu_check.json").read_text());a=json.loads((X/"dev46_gpu_archive.json").read_text())
    if not q["passed"] or not a["all_members_roundtrip_verified"]:raise RuntimeError("incomplete")
    timing=[]
    for b in q["benchmarks"]:
        old=statistics.median(v["seconds_per_call"] for v in b["groups"] if v["method"]=="original")
        new=statistics.median(v["seconds_per_call"] for v in b["groups"] if v["method"]=="reused_probability")
        timing.append({"label":b["label"],"original_pointwise_seconds":old,"reused_probability_seconds":new,"ratio":new/old,"setup":b["setup"]})
    reps=[r for c in q["cells"] for r in c["repetitions"]]
    result={"task_id":"TRR-0020","scope":q["scope"],"cases":len(q["cells"]),"full_update_repetitions":len(reps),
      "timed_graph_calls":sum(g["calls"] for b in q["benchmarks"] for g in b["groups"]),
      "all_update_bytes_equal":all(v["all_bytes_equal"] for v in reps),
      "all_graph_bytes_equal":all(g["all_bytes_equal"] for b in q["benchmarks"] for g in b["groups"]),
      "all_inputs_and_probabilities_unchanged":all(c["inputs_unchanged"] and c["probability_unchanged"] for c in q["cells"]),
      "timings":timing,"peak_reserved_GiB":q["peak_reserved"]/2**30,"seconds":q["end_unix"]-q["start_unix"],
      "archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],"new_archive_bytes":a["new_archive_bytes"],
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev46_gpu_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

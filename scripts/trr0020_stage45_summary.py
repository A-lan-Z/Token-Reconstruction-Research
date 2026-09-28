"""Describe complete exact pointwise decoder evidence without replacement claims."""
from pathlib import Path
import json,time,subprocess,sys,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    f=json.loads((X/"dev45_freeze.json").read_text());s=json.loads((X/"dev45_score.json").read_text())
    g=json.loads((X/"dev45_guard.json").read_text());a=json.loads((X/"dev45_archive.json").read_text())
    if g["returncode"]!=0 or g["failure"] is not None or s["freeze_sha256"]!=sha(X/"dev45_freeze.json"):raise RuntimeError("incomplete run")
    groups=s["summary"];primary=[v for v in groups if v["variant"].startswith("step")]
    timing=[]
    for mode in ["control","exact"]:
      for steps in [64,128]:
        for length in [40,128]:
            values=[v for v in f["entries"] if v["mode"]==mode and v["steps"]==steps and v["positions"]==length]
            timing.append({"mode":mode,"steps":steps,"length":length,"records":len(values),
              "mean_median_seconds":statistics.mean(v["median_seconds"] for v in values),
              "mean_lookup_seconds":statistics.mean(statistics.median(r["stats"]["first_lookup_seconds"] for r in v["repetitions"]) for v in values)})
    phases=[];allq=[]
    for entry in f["phases"]:
        p=json.loads((ROOT/entry["path"]).read_text());q=json.loads((ROOT/p["qualification_path"]).read_text());allq.append(q)
        phases.append({k:p[k] for k in ["mode","prefix_seconds","engine_seconds","warm_capture_events","mirror_capture_events","peak_reserved","peak_host_rss","execution_commit","start_unix","end_unix"]})
    if not all(v["anchor_valid"] and v["exact_repeat"] for e in f["entries"] for v in e["repetitions"]):raise RuntimeError("nonexact reconstruction")
    for e in f["entries"]:
        if e["mode"]!="exact":continue
        reference=next(v for v in f["entries"] if v["id"]==e["id"] and v["steps"]==e["steps"] and v["mode"]=="control")
        if e["sha256"]!=reference["sha256"]:raise RuntimeError("paired saved output bytes differ")
    result={"task_id":"TRR-0020","all_complete_outputs_and_recorded_traces_exact":True,
      "real_original_anchor_checks":sum(len(e["repetitions"]) for e in f["entries"]),"scope":s["scope"],"reconstruction_cells":32,"repetitions":96,
      "primary_groups":primary,"all_readout_groups":groups,"timings":timing,"phases":phases,
      "public_repeated_decodes":sum(len(q["runs"]) for q in allq),"eager_checks":sum(len(q["eager"]) for q in allq),
      "independent_gradient_checks":sum(len(q["gradients"]) for q in allq),
      "maximum_gradient_error":max(v["maximum_gradient_error"] for q in allq for v in q["gradients"]),
      "public_complete_original_anchors":sum(len(q["runs"]) for q in allq),
      "peak_reserved_GiB":max(p["peak_reserved"] for p in phases)/2**30,
      "guard_seconds":g["end_unix"]-g["start_unix"],"minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),
      "maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],"archive_bytes":a["raw_archive_bytes"],
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev45_summary.json").open("x") as out:json.dump(result,out,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ["all_readout_groups","phases"]},indent=2))
if __name__=="__main__":main()

"""Reproducible descriptive summary of the complete cold probability-budget study."""
from pathlib import Path
import json,statistics,time
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    freeze=json.loads((X/"dev32_grid_freeze.json").read_text());score=json.loads((X/"dev32_grid_score.json").read_text())
    guard=json.loads((X/"dev32_grid_guard.json").read_text());archive=json.loads((X/"dev32_grid_archive.json").read_text())
    configurations=[]
    for name,factor,iterations in freeze["configurations"]:
        entries=[e for e in freeze["entries"] if e["method"]==name];timings=[]
        for length in [40,128]:
            values=[e for e in entries if e["positions"]==length]
            timings.append({"positions":length,"records":len(values),
              "mean_total_seconds":statistics.mean(e["stats"]["total_seconds"] for e in values),
              "mean_warm_seconds":statistics.mean(e["stats"]["warm_stage_seconds_including_capture"] for e in values),
              "mean_mirror_seconds":statistics.mean(e["stats"]["mirror_seconds"] for e in values),
              "checkpoint_observed_seconds":{str(k):statistics.mean(e["stats"]["checkpoint_total_seconds"][str(k)] for e in values) for k in [0,1,2,4,8,16,32,64]}})
        groups=[v for v in score["summary"] if v["method"]==name and not v["variant"].startswith("warm_")]
        configurations.append({"method":name,"factor":factor,"scalar_iterations":iterations,"groups":groups,"timings":timings})
    qualifications=[]
    for ref in freeze["phases"]:
        p=json.loads((ROOT/ref["path"]).read_text());qualifications.append(json.loads((ROOT/p["qualification_path"]).read_text()))
    result={"task_id":"TRR-0020","scope":"retrospective eight-record development; no new canonical comparison",
      "cells":len(freeze["entries"]),"configurations":configurations,
      "warm_anchors":sum(e["warm_anchor"]["equal"] for e in freeze["entries"]),
      "mean_anchors":sum(e["initial_mean_anchor"]["equal"] for e in freeze["entries"]),
      "public_repeat_decodes":sum(len(q["runs"]) for q in qualifications),"public_eager_checks":sum(len(q["eager"]) for q in qualifications),
      "first_step_public_anchors":sum(len(q["one_step_anchors"]) for q in qualifications),
      "direct_kl_checks":sum(len(v["checks"]) for q in qualifications for v in q["kl_references"]),
      "max_direct_kl_budget_excess":max(max(a-b for a,b in zip(c["direct_kl"],c["budget"])) for q in qualifications for v in q["kl_references"] for c in v["checks"]),
      "gradient_checks":sum(len(q["gradient_reference"]) for q in qualifications),
      "gradient_max_absolute_error":max(v["max_absolute_error"] for q in qualifications for v in q["gradient_reference"]),
      "peak_reserved_GiB":max(q["peak_reserved"] for q in qualifications)/2**30,
      "guard_returncode":guard["returncode"],"guard_failure":guard["failure"],
      "guard_seconds":guard["end_unix"]-guard["start_unix"],"minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in guard["samples"]),
      "maximum_temperature_c":max(v["temperature_c"] for v in guard["samples"]),
      "archive_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],
      "archive_bytes":archive["raw_archive_bytes"],"archive_parts":len(archive["parts"]),"created_unix":time.time()}
    p=X/"dev32_grid_summary.json"
    if p.exists():raise RuntimeError("summary exists")
    p.write_text(json.dumps(result,indent=2)+"\n")
    for cfg in configurations:
        print(cfg["method"],[(v["positions"],round(v["mean_total_seconds"],4)) for v in cfg["timings"]])
        for variant in ["step1","step2","step4","step8","step16","step32","step64","best_objective","best_position_error"]:
            print(variant,[(v["condition"],v["group"],v["correct"],v["scored"],v["exact"]) for v in cfg["groups"] if v["variant"]==variant])
    print(json.dumps({k:v for k,v in result.items() if k!="configurations"},indent=2))
if __name__=="__main__":main()

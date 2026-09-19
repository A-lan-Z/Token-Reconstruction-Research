"""Summarize the frozen committed-context study without changing decisions."""
from pathlib import Path
import json,statistics,time
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    freeze=json.loads((X/"dev27_decoder_freeze.json").read_text())
    score=json.loads((X/"dev27_decoder_score.json").read_text())
    guard=json.loads((X/"dev27_decoder_guard.json").read_text())
    archive=json.loads((X/"dev27_decoder_archive.json").read_text())
    configs=[]
    for name,warm,rule,updates in freeze["configurations"]:
        entries=[v for v in freeze["entries"] if v["method"]==name]
        timings=[]
        for length in [40,128]:
            values=[v for v in entries if v["positions"]==length]
            keys=set().union(*(v["stats"]["phase_seconds"] for v in values))
            timings.append({"positions":length,"records":len(values),
              "mean_total_seconds":statistics.mean(v["stats"]["total_seconds"] for v in values),
              "min_total_seconds":min(v["stats"]["total_seconds"] for v in values),
              "max_total_seconds":max(v["stats"]["total_seconds"] for v in values),
              "mean_phase_seconds":{k:statistics.mean(v["stats"]["phase_seconds"].get(k,0) for v in values) for k in sorted(keys)}})
        groups=[v for v in score["summary"] if v["method"]==name and v["variant"] in ["tokens","warm_step"+str(warm)]]
        configs.append({"method":name,"warm":warm,"rule":rule,"updates":updates,"groups":groups,"timings":timings})
    qualifications=[]
    for p in freeze["phases"]:
        phase=json.loads((ROOT/p["path"]).read_text())
        qualifications.append(json.loads((ROOT/phase["qualification_path"]).read_text()))
    summary={"task_id":"TRR-0020","scope":"retrospective eight-record development; not canonical comparison",
      "cells":len(freeze["entries"]),"configurations":configs,
      "controls":{"warm_anchors":sum(e["warm_anchor"]["equal"] for e in freeze["entries"]),
        "mean_anchors":sum(e["initial_mean_anchor"]["equal"] for e in freeze["entries"]),
        "exact_public_runs":sum(len(q["runs"]) for q in qualifications),
        "causal_references":sum(len(q["causal_reference"]) for q in qualifications),
        "readout_references":sum(len(q["score_reference"]) for q in qualifications),
        "all_qualifications_passed":all(q["passed"] for q in qualifications),
        "max_causal_reference_absolute_error":max(v["max_absolute_error"] for q in qualifications for ref in q["causal_reference"] for v in ref["checks"])},
      "peak_reserved_GiB":max(q["peak_reserved"] for q in qualifications)/2**30,
      "guard_returncode":guard["returncode"],"guard_failure":guard["failure"],
      "guard_seconds":guard["end_unix"]-guard["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in guard["samples"]),
      "maximum_temperature_c":max(v["temperature_c"] for v in guard["samples"]),
      "archive_logical_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],
      "archive_bytes":archive["raw_archive_bytes"],"archive_parts":len(archive["parts"]),
      "created_unix":time.time()}
    dest=X/"dev27_decoder_summary.json"
    if dest.exists():raise RuntimeError("create-only summary")
    dest.write_text(json.dumps(summary,indent=2)+"\n")
    for cfg in configs:
        print(cfg["method"],[(v["condition"],v["group"],v["correct"],v["scored"],v["exact"]) for v in cfg["groups"] if v["variant"]=="tokens"],
          [(v["positions"],round(v["mean_total_seconds"],4)) for v in cfg["timings"]])
    print(json.dumps({k:v for k,v in summary.items() if k!="configurations"},indent=2))
if __name__=="__main__":main()

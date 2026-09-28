"""Summarize the complete development33 causal full-vocabulary decoder study."""
from pathlib import Path
import json,statistics,time,hashlib
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def read(name):return json.loads((X/name).read_text())
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    f=read("dev33_decoder_freeze.json");score=read("dev33_decoder_score.json")
    guard=read("dev33_decoder_guard.json");archive=read("dev33_decoder_archive.json")
    if guard["returncode"]!=0 or guard["failure"] is not None:raise RuntimeError("guard not successful")
    if score["freeze_sha256"]!=sha(X/"dev33_decoder_freeze.json"):raise RuntimeError("score binding changed")
    configurations=[]
    for name,factor,steps in f["configurations"]:
        entries=[e for e in f["entries"] if e["method"]==name];timings=[]
        for length in [40,128]:
            values=[e for e in entries if e["positions"]==length]
            timings.append({"positions":length,"records":len(values),
              "mean_total_seconds":statistics.mean(e["stats"]["total_seconds"] for e in values),
              "minimum_total_seconds":min(e["stats"]["total_seconds"] for e in values),
              "maximum_total_seconds":max(e["stats"]["total_seconds"] for e in values),
              "mean_io_hash_seconds":statistics.mean(e["io_hash_seconds"] for e in values),
              "per_condition":[{"condition":condition,"records":sum(e["condition"]==condition for e in values),
                "mean_total_seconds":statistics.mean(e["stats"]["total_seconds"] for e in values if e["condition"]==condition)}
                for condition in ["matched","lora256"]]})
        configurations.append({"method":name,"factor":factor,"steps":steps,
          "groups":[v for v in score["summary"] if v["method"]==name],"timings":timings,
          "counts_by_length":[{k:e["stats"][k] for k in ["positions","prefix_forwards","prefix_vjps",
             "history_commit_forwards","mixture_products","probability_gradient_products","initialization_vocabulary_products"]}
             for length in [40,128] for e in entries if e["id"]==next(v["id"] for v in entries if v["positions"]==length)]})
    phases=[json.loads((ROOT/ref["path"]).read_text()) for ref in f["phases"]]
    qs=[json.loads((ROOT/p["qualification_path"]).read_text()) for p in phases]
    result={"task_id":"TRR-0020","scope":"retrospective eight-record development panel; no canonical method selected",
      "cells":len(f["entries"]),"configurations":configurations,
      "public_repeat_decodes":sum(len(q["runs"]) for q in qs),
      "public_eager_checks":sum(len(q["eager"]) for q in qs),
      "public_emitted_history_checks":sum(len(q["history_references"]) for q in qs),
      "all_public_outputs_exact":all(v["exact_repeat"] for q in qs for v in q["runs"]+q["eager"]),
      "all_history_checks_passed":all(v["passed"] for q in qs for v in q["history_references"]),
      "capture_geometries_per_worker":[len(q["capture_events"]) for q in qs],
      "preparation":[{k:p[k] for k in ["factor","prefix_seconds","engine_seconds","capture_seconds",
        "peak_reserved","peak_allocated","peak_host_rss","code_commit"]} for p in phases],
      "preparation_scope":"Each factor worker loads public prefix, builds tables and captures all128 native contexts; shared across its four update-count methods. These costs are separate from per-record decode time.",
      "per_record_timing_scope":f["entries"][0]["stats"]["timing_scope"],
      "peak_reserved_GiB":max(p["peak_reserved"] for p in phases)/2**30,
      "guard_returncode":guard["returncode"],"guard_failure":guard["failure"],
      "guard_seconds":guard["end_unix"]-guard["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in guard["samples"]),
      "maximum_temperature_c":max(v["temperature_c"] for v in guard["samples"]),
      "archive_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],
      "archive_bytes":archive["raw_archive_bytes"],"archive_parts":len(archive["parts"]),
      "freeze_sha256":sha(X/"dev33_decoder_freeze.json"),"score_sha256":sha(X/"dev33_decoder_score.json"),
      "created_unix":time.time()}
    with (X/"dev33_decoder_summary.json").open("x") as out:json.dump(result,out,indent=2);out.write("\n")
    for cfg in configurations:
        print(cfg["method"],[(v["condition"],v["group"],v["correct"],v["scored"],v["exact"]) for v in cfg["groups"]],
          [(v["positions"],round(v["mean_total_seconds"],4)) for v in cfg["timings"]])
    print(json.dumps({k:v for k,v in result.items() if k!="configurations"},indent=2))
if __name__=="__main__":main()

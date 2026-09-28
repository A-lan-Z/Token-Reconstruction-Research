"""Paired public comparisons of ordinary and norm-preserving embedding mixtures."""
from pathlib import Path
import json,time,statistics
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev35_public_probe.json").read_text());g=json.loads((X/"dev35_guard.json").read_text())
    a=json.loads((X/"dev35_public_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or g["failure"] is not None:raise RuntimeError("incomplete public study")
    rows=[]
    for c in q["cells"]:
        context=next(v for v in q["contexts"] if v["precision"]==c["precision"] and v["context"]==c["context"])
        reps=c["repetitions"];first=reps[0]
        row={k:c[k] for k in ["precision","context","method","rule","steps"]}
        row.update(public_text=context["public_text"],raw_initial_error=context["raw_initial_error"],normal_initial_error=context["normal_initial_error"],
          raw_norm=context["raw_norm"],expected_norm=context["expected_norm"],initial_scale=context["initial_scale"],
          observed_error=first["observed_error"],confidence=first["confidence"],
          identified={name:first[key]==context["public_token_id"] for name,key in [("probability","emitted_token_id"),("euclidean","euclidean_token_id"),("metric","metric_token_id")]},
          prefix_forwards=first["prefix_forwards"],prefix_vjps=first["prefix_vjps"],
          median_inference_seconds=statistics.median(v["inference_seconds"] for v in reps),
          median_readout_seconds=statistics.median(v["final_readout_seconds"] for v in reps),
          median_commit_seconds=statistics.median(v["emitted_token_commit_seconds"] for v in reps))
        rows.append(row)
    result={"task_id":"TRR-0020","scope":"public synthetic common-token numerical sanity; no benchmark reconstruction accuracy",
      "cells":len(q["cells"]),"repetitions":sum(len(c["repetitions"]) for c in q["cells"]),
      "gradient_references":len(q["gradient_references"]),"gradient_max_error":max(v["gradient_max_error"] for v in q["gradient_references"]),
      "one_hot_checks":len(q["one_hot_checks"]),"one_hot_max_error":max(v["max_error"] for v in q["one_hot_checks"]),
      "unchanged_control_anchors":len(q["control_anchors"]),"public_identity_sanity":q["public_identity_sanity"],"rows":rows,
      "initial_scale_range":[min(v["initial_scale"] for v in q["contexts"]),max(v["initial_scale"] for v in q["contexts"])],
      "norm_table_seconds":q["norm_table_seconds"],"norm_table_bytes":q["norm_table_bytes"],
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "archive_members":a["logical_member_count"],"archive_unique_objects":a["unique_objects"],"archive_bytes":a["raw_archive_bytes"],"created_unix":time.time()}
    with (X/"dev35_public_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    for row in rows:
        if row["rule"]=="normalized" and (row["public_text"] in ["a","the","The","("] or not all(row["identified"].values())):
            print(json.dumps(row))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()

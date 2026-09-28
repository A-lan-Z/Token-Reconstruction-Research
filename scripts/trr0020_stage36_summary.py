"""Summarize public endpoint and probability-path geometry without reconstruction claims."""
from pathlib import Path
import json,time,collections
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def main():
    q=json.loads((X/"dev36_public_curve.json").read_text());g=json.loads((X/"dev36_guard.json").read_text())
    archive=json.loads((X/"dev36_public_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or g["failure"] is not None:raise RuntimeError("incomplete diagnostic")
    rows=[];groups=collections.defaultdict(list)
    for v in q["curves"]:
        c=v["cosine_errors"];m=v["normalized_squared_errors"]
        row={k:v[k] for k in ["precision","context","public_text","mode","initial_true_probability","initial_true_logit_rank","initial_true_gradient_rank","initial_cosine_slope","jvp_cosine_slope","initial_squared_error_slope"]}
        row.update(initial_cosine_error=c[0],endpoint_cosine_error=c[-1],maximum_cosine_increase=max(c)-c[0],
          maximum_cosine_alpha=q["alphas"][c.index(max(c))],initial_squared_error=m[0],endpoint_squared_error=m[-1],
          maximum_squared_error_increase=max(m)-m[0],maximum_squared_error_alpha=q["alphas"][m.index(max(m))])
        rows.append(row);groups[row["precision"],row["mode"]].append(row)
    grouped=[]
    for key,values in groups.items():
        grouped.append({"precision":key[0],"mode":key[1],"contexts":len(values),
          "positive_initial_cosine_slope":sum(v["initial_cosine_slope"]>1e-6 for v in values),
          "positive_initial_squared_error_slope":sum(v["initial_squared_error_slope"]>1e-6 for v in values),
          "cosine_path_increase_above_1e_6":sum(v["maximum_cosine_increase"]>1e-6 for v in values),
          "squared_error_path_increase_above_1e_6":sum(v["maximum_squared_error_increase"]>1e-6 for v in values),
          "cosine_up_squared_error_down":sum(v["initial_cosine_slope"]>1e-6 and v["initial_squared_error_slope"]< -1e-6 for v in values)})
    result={"task_id":"TRR-0020","scope":q["scope"],"curves":len(rows),"rows":rows,"groups":grouped,
      "endpoint_checks":len(q["input_anchors"]),"maximum_native_vs_full_error":max(v["native_vs_full_max_error"] for v in q["input_anchors"]),
      "directional_derivative_checks":len(q["curves"]),"maximum_directional_derivative_discrepancy":max(abs(v["initial_cosine_slope"]-v["jvp_cosine_slope"]) for v in q["curves"]),
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "archive_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],"archive_bytes":archive["raw_archive_bytes"],
      "created_unix":time.time()}
    with (X/"dev36_public_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    for row in rows:
        if row["public_text"] in ["a","the"]:print(json.dumps(row))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()

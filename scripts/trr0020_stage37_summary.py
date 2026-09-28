"""Summarize frozen known-public direct-score checks; no benchmark claims."""
from pathlib import Path
import json,time,hashlib,statistics,subprocess,sys
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(entry):
    if digest(ROOT/entry["path"])!=entry["sha256"]:raise RuntimeError("frozen artifact changed")
    return load_file(str(ROOT/entry["path"]))
def main():
    torch.set_num_threads(2);start=time.time()
    q=json.loads((X/"dev37_public_probe.json").read_text());g=json.loads((X/"dev37_guard.json").read_text())
    archive=json.loads((X/"dev37_public_archive.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or g["failure"] is not None:raise RuntimeError("incomplete probe")
    if len(q["cells"])!=16 or any(len(v["repetitions"])!=3 or not all(e["exact_repeat"] for e in v["repetitions"]) for v in q["cells"]):
        raise RuntimeError("incomplete repeats")
    for path,sha in q["sources"].items():
        if digest(ROOT/path)!=sha:raise RuntimeError("source changed")
    fixture={r["past"]:read(r) for r in q["contexts"]}
    rows=[];groups=[]
    for cell in q["cells"]:
        data=read(cell["repetitions"][0]);ids=fixture[cell["past"]]["public_ids"]
        reps=cell["repetitions"]
        cost=[v["correction_seconds"]+v["norm_adjustment_seconds"]+statistics.median(v["per_observation_scoring_seconds"]) for v in reps]
        for index,metric in enumerate(q["metrics"]):
            s=data[metric];emitted=s.argmax(-1)
            if s.shape!=(16,128256) or not torch.equal(emitted,data["tokens"][:,index]):raise RuntimeError("not full-vocabulary argmax")
            true_score=s.gather(1,ids[:,None])[:,0];rank=1+(s>true_score[:,None]).sum(-1);ok=emitted==ids
            rows.extend({"past":cell["past"],"mode":cell["mode"],"metric":metric,
               "public_text":text,"public_id":int(actual),"emitted_id":int(pred),"correct":bool(correct),"true_score_rank":int(r)}
               for text,actual,pred,correct,r in zip(q["public_texts"],ids,emitted,ok,rank))
            groups.append({"past":cell["past"],"mode":cell["mode"],"metric":metric,"public_correct":int(ok.sum()),
              "public_cases":16,"median_true_rank":float(rank.double().median()),"maximum_true_rank":int(rank.max()),
              "median_probe_seconds":statistics.median(v["correction_seconds"] for v in reps),
              "median_norm_seconds":statistics.median(v["norm_adjustment_seconds"] for v in reps),
              "median_pair_score_seconds":statistics.median(statistics.median(v["per_observation_scoring_seconds"]) for v in reps),
              "median_pair_score_with_context_correction_seconds":statistics.median(cost),
              "prefix_probe_forwards":reps[0]["correction_prefix_forwards"],
              "norm_fallback_rows":reps[0]["numerical_norm_fallback_rows"]})
    result={"task_id":"TRR-0020","scope":q["scope"],"public_decisions":len(rows),"repeated_decisions":len(rows)*3,
      "rows":rows,"groups":groups,"adapter_checks":len(q["adapter_checks"]),"cache_anchor_rows":512,
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),
      "maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "preparation":q["preparation"],"batch_geometry_diagnostic":[r for r in q["contexts"] if r["past"]==1],
      "archive_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],
      "archive_bytes":archive["raw_archive_bytes"],
      "cost_scope":"both score rules computed together; per-position correction charged in full; emitted-token commit and record transfers absent; not end-to-end runtime",
      "passed":True,"source_receipt_sha256":digest(X/"dev37_public_probe.json"),
      "source_summary_code_sha256":digest(Path(__file__)),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":start,"end_unix":time.time()}
    with (X/"dev37_public_summary.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()

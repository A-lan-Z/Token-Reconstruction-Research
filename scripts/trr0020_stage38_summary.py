"""Summarize only the complete frozen public first-token panel."""
from pathlib import Path
import json,hashlib,time,subprocess,sys,statistics
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    torch.set_num_threads(2);q=json.loads((X/"dev38_public_probe.json").read_text());g=json.loads((X/"dev38_public_guard.json").read_text())
    if not q["passed"] or g["returncode"]!=0 or g["failure"] is not None or len(q["families"])!=4:raise RuntimeError("incomplete public matrix")
    for path,h in q["sources"].items():
        if sha(ROOT/path)!=h:raise RuntimeError("source changed")
    rows=[]
    for entry in q["families"]:
        if len(entry["repetitions"])!=3 or not all(v["exact_repeat"] for v in entry["repetitions"]):raise RuntimeError("repeat qualification")
        if sha(ROOT/entry["path"])!=entry["sha256"]:raise RuntimeError("output changed")
        data=load_file(str(ROOT/entry["path"]));scores=data["scores"];target=data["public_ids"]
        if scores.shape!=(entry["count"],128256) or not torch.equal(data["tokens"],scores.argmax(-1)):raise RuntimeError("invalid full-vocabulary decision")
        correct=data["tokens"]==target;true_score=scores.gather(1,target[:,None]);ranks=1+(scores>true_score).sum(-1)
        rows.append({"precision":entry["precision"],"family":entry["family"],"public_correct":int(correct.sum()),"count":len(target),
          "maximum_true_rank":int(ranks.max()),"median_seconds":statistics.median(t for r in entry["repetitions"] for t in r["score_seconds"]),
          "wrong_public_ids":target[~correct].tolist(),"emitted_wrong_ids":data["tokens"][~correct].tolist()})
    out={"task_id":"TRR-0020","scope":q["scope"],"groups":rows,"public_decisions":288,"repetitions":864,
      "independent_score_checks":len(q["score_references"]),"maximum_score_error":max(v["maximum_score_error"] for v in q["score_references"]),
      "peak_reserved_GiB":q["peak_reserved"]/2**30,"guard_seconds":g["end_unix"]-g["start_unix"],
      "minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in g["samples"]),"maximum_temperature_c":max(v["temperature_c"] for v in g["samples"]),
      "preparation":q["preparation"],"source_receipt_sha256":sha(X/"dev38_public_probe.json"),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev38_public_summary.json").open("x") as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()

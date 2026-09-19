"""Post-freeze confidence diagnostic; never changes a submitted reconstruction."""
from pathlib import Path
import json,hashlib,collections,torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def digest(p):
    with Path(p).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def describe(values):
    if not values:return {"count":0}
    a=torch.cat(values).double()
    if not len(a):return {"count":0}
    return {"count":len(a),"mean":float(a.mean()),"median":float(a.median()),"min":float(a.min()),"max":float(a.max()),
      "below_half":int((a<.5).sum()),"above_0p9":int((a>.9).sum()),"above_0p99":int((a>.99).sum())}
def main():
    truthpath=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1/evaluator_truth.json"
    records=[]
    for stage in [29,30]:
        freeze=json.loads((X/f"dev{stage}_freeze.json").read_text())
        scored=json.loads((X/f"dev{stage}_score.json").read_text())
        if scored["truth_sha256"]!=digest(truthpath) or scored["freeze_sha256"]!=digest(X/f"dev{stage}_freeze.json"):raise RuntimeError("changed scored source")
        truth=json.loads(truthpath.read_text());groups=collections.defaultdict(lambda:{"correct":[],"wrong":[],"wrong_error":[],"correct_error":[]})
        for entry in freeze["entries"]:
            if digest(ROOT/entry["path"])!=entry["sha256"]:raise RuntimeError("changed prediction")
            data=load_file(str(ROOT/entry["path"]));target=torch.tensor(truth[entry["id"]]);correct=data["step64"][1:]==target[1:]
            group=groups[entry["method"],entry["condition"],entry["group"]]
            group["correct"].append(data["final_confidence"][correct]);group["wrong"].append(data["final_confidence"][~correct])
            group["wrong_error"].append(data["final_position_error"][~correct]);group["correct_error"].append(data["final_position_error"][correct])
        for key,values in groups.items():
            row={"stage":stage,"method":key[0],"condition":key[1],"group":key[2],
              "correct_confidence":describe(values["correct"]),"wrong_confidence":describe(values["wrong"]),
              "correct_observed_error":describe(values["correct_error"]),"wrong_observed_error":describe(values["wrong_error"])}
            records.append(row)
    result={"task_id":"TRR-0020","scope":"post-freeze retrospective diagnostic of final step64 only; labels used only for grouping; no output changes",
      "source_scores":{str(stage):digest(X/f"dev{stage}_score.json") for stage in [29,30]},"rows":records}
    p=X/"dev29_30_confidence_diagnostic.json"
    if p.exists():raise RuntimeError("diagnostic exists")
    p.write_text(json.dumps(result,indent=2)+"\n")
    for row in records:
        if row["method"] in ["warm0_tau0.01","warm64_tau0.01","warm64_tau0.01_error"]:
            print(row["stage"],row["method"],row["condition"],row["group"],"WRONG",row["wrong_confidence"],"WRONG_ERROR",row["wrong_observed_error"])
if __name__=="__main__":main()

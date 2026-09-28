"""Post-freeze causal error audit; never changes reconstruction outputs."""
from pathlib import Path
import sys,os,json,time,hashlib,statistics,subprocess,collections
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage33_decoder"))
from score import gate
from support import INPUT
from safetensors.torch import load_file
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def describe(values):
    if not values:return {"count":0}
    return {"count":len(values),"minimum":min(values),"median":statistics.median(values),"maximum":max(values)}
def main():
    started=time.time();freeze_path=X/"dev33_decoder_freeze.json";f=json.loads(freeze_path.read_text())
    existing_score=json.loads((X/"dev33_decoder_score.json").read_text())
    guard=json.loads((X/"dev33_decoder_guard.json").read_text())
    if guard["returncode"]!=0 or guard["failure"] is not None:raise RuntimeError("guard not successful")
    outputs=gate(f)
    if existing_score["freeze_sha256"]!=sha(freeze_path):raise RuntimeError("changed freeze")
    truth_path=INPUT/"evaluator_truth.json"
    if existing_score["truth_sha256"]!=sha(truth_path):raise RuntimeError("changed scored truth")
    bound_unix=time.time();truth=json.loads(truth_path.read_text());rows=[];groups=collections.defaultdict(list)
    scores={(e["id"],e["method"]):e for e in existing_score["rows"]}
    for e in f["entries"]:
        key=e["id"],e["method"];data=load_file(str(ROOT/e["path"]),device="cpu")
        ids=outputs[key].tolist();target=truth[e["id"]];steps=e["stats"]["steps"];mistakes=[];correct=[]
        if len(ids)!=len(target):raise RuntimeError("length mismatch")
        for pos in range(1,len(ids)):
            trace=data["update_trace"][pos-1,:steps]
            loss=[float(v) for v in trace[:,0]]+[float(data["soft_error"][pos])]
            confidence=float(data["confidence"][pos])
            row={"position":pos,"final_soft_error":loss[-1],"final_confidence":confidence,
                 "initial_soft_error":loss[0],"relative_soft_error":loss[-1]/loss[0] if loss[0]>0 else None,
                 "update_losses":loss,"loss_increase_count":sum(b>a+1e-7 for a,b in zip(loss,loss[1:])),
                 "minimum_budget_used":float(trace[:,4].min())}
            (correct if ids[pos]==target[pos] else mistakes).append(row)
        if len(correct)!=scores[key]["correct"] or len(correct)+len(mistakes)!=scores[key]["scored"]:raise RuntimeError("score disagreement")
        row={k:e[k] for k in ["id","method","condition","group","positions"]}
        row.update(steps=steps,correct=len(correct),scored=len(ids)-1,first_error=mistakes[0] if mistakes else None,
          subsequent_errors=mistakes[1:],correct_positions=correct,prediction_path=e["path"],prediction_sha256=e["sha256"])
        rows.append(row);groups[row["method"],row["condition"],row["group"]].append(row)
    summary=[]
    for key,values in groups.items():
        first=[v["first_error"] for v in values if v["first_error"] is not None]
        later=[e for v in values for e in v["subsequent_errors"]]
        correct=[e for v in values for e in v["correct_positions"]]
        row=dict(zip(["method","condition","group"],key))
        row.update(records=len(values),first_error_positions=describe([e["position"] for e in first]))
        for name,entries in [("first_errors",first),("subsequent_errors",later),("correct_positions",correct)]:
            row[name]={"count":len(entries),"final_confidence":describe([e["final_confidence"] for e in entries]),
              "final_soft_error":describe([e["final_soft_error"] for e in entries]),
              "initial_soft_error":describe([e["initial_soft_error"] for e in entries]),
              "relative_soft_error":describe([e["relative_soft_error"] for e in entries if e["relative_soft_error"] is not None]),
              "loss_increases":sum(e["loss_increase_count"] for e in entries)}
        summary.append(row)
    result={"task_id":"TRR-0020","scope":"retrospective descriptive analysis of previously scored frozen outputs; no reconstruction changes",
      "freeze_sha256":sha(freeze_path),"score_sha256":sha(X/"dev33_decoder_score.json"),
      "truth_sha256":sha(truth_path),"binding_completed_unix":bound_unix,"records":len(rows),"rows":rows,"summary":summary,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "script_sha256":sha(Path(__file__)),"command":[sys.executable,*sys.argv],
      "environment_flags":{k:os.environ.get(k) for k in ["PYTHONPATH","CUBLAS_WORKSPACE_CONFIG","PYTHONDONTWRITEBYTECODE"]},
      "start_unix":started,"end_unix":time.time(),"outputs_changed":False}
    with (X/"dev34_causal_error_audit.json").open("x") as out:json.dump(result,out,indent=2);out.write("\n")
    for row in rows:
        if row["method"]=="causal_factor2.0_steps8":print(json.dumps({k:row[k] for k in ["id","method","correct","scored","first_error"]}))
    print("COMPLETE",len(rows),"records")
if __name__=="__main__":main()

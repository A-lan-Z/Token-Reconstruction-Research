"""Open retrospective truth only after complete matrix and qualification gates."""
from support import *
import copy,collections
def gate(f):
    expected={(r["id"],name) for r in rows() for name,_,_ in CONFIGS}
    cells=[(v["id"],v["method"]) for v in f["entries"]]
    if len(cells)!=48 or len(set(cells))!=48 or set(cells)!=expected:raise ValueError("incomplete matrix")
    if f["binding"]!=binding() or len(f["phases"])!=2:raise ValueError("changed source or phases")
    for phase in f["phases"]:
        if n.digest(ROOT/phase["path"])!=phase["sha256"]:raise ValueError("phase hash")
        p=json.loads((ROOT/phase["path"]).read_text())
        if n.digest(ROOT/p["qualification_path"])!=p["qualification_sha256"]:raise ValueError("qualification hash")
        q=json.loads((ROOT/p["qualification_path"]).read_text())
        if not q["passed"] or len(q["runs"])!=18 or len(q["eager"])!=6 or len(q["gradients"])!=2:raise ValueError("incomplete public qualification")
        if not all(v["passed"] for v in q["gradients"]):raise ValueError("gradient qualification")
        for v in q["runs"]:
            if not all(v[k] for k in ["exact_repeat","fixed_position_valid","anchor_valid"]) or not v["stats"]["numeric_valid"]:raise ValueError("invalid repeat")
        for v in q["eager"]:
            if not v["exact_equal"] or not v["fixed_position_valid"]:raise ValueError("invalid eager check")
        for v in q["runs"]+q["eager"]:
            if n.digest(ROOT/v["path"])!=v["sha256"]:raise ValueError("changed public output")
    outputs={}
    for entry in f["entries"]:
        data=verify(entry,f["binding"]);length=entry["positions"];stats=entry["stats"];steps=entry["steps"]
        if len(entry["repetitions"])!=3 or not all(r["fixed_position_valid"] and r["anchor_valid"] and r["stats"]["numeric_valid"] for r in entry["repetitions"]):raise ValueError("invalid real repeats")
        if stats["shortlist_size"] is not None or stats["separate_candidate_verification_calls"]!=0 or stats["model_parameter_updates"]!=0 or stats["vocabulary_entries_per_sweep"]!=128256:raise ValueError("method scope")
        if stats["whole_sequence_prefix_forwards"]!=steps+2 or stats["whole_sequence_prefix_backwards"]!=steps:raise ValueError("work accounting")
        if stats["factor"]!=2. or stats["scalar_iterations"]!=4 or stats["first_locked"]!=(entry["mode"]=="locked"):raise ValueError("rule changed")
        if n.digest(ROOT/entry["old_anchor"]["path"])!=entry["old_anchor"]["sha256"]:raise ValueError("anchor changed")
        for key,value in data.items():
            if not torch.isfinite(value).all():raise ValueError("nonfinite output")
            if key not in AUX:
                if value.shape!=(length,) or value.dtype!=torch.int64 or int(value[0])!=128000 or bool((value<0).any() or (value>=128256).any()):raise ValueError("token geometry")
        if entry["mode"]=="locked":
            for key in ["step"+str(steps),"best_objective","best_position_error"]:
                if not torch.equal(data[key][1:2],data["first_choice"]):raise ValueError("fixed first decision changed")
        outputs[entry["id"],entry["method"]]=data
    return outputs
def main():
    f=json.loads((X/"dev38_grid_freeze.json").read_text());outputs=gate(f);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(f)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["table_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("negative gate failed")
    n.write(X/"dev38_grid_truth_gate.json",{"cells":48,"rejected":negative,"freeze_sha256":n.digest(X/"dev38_grid_freeze.json"),"truth_opened_after_unix":time.time()})
    truthpath=INPUT/"evaluator_truth.json";truth=json.loads(truthpath.read_text());result=[];groups=collections.defaultdict(list)
    for entry in f["entries"]:
        target=torch.tensor(truth[entry["id"]]);data=outputs[entry["id"],entry["method"]]
        for variant in ["step"+str(entry["steps"]),"best_objective","best_position_error"]:
            ids=data[variant]
            if ids.shape!=target.shape:raise RuntimeError("truth geometry")
            ok=ids[1:]==target[1:]
            row={k:entry[k] for k in ["id","condition","group","method","mode","steps"]}
            row.update(variant=variant,correct=int(ok.sum()),scored=len(ok),exact=bool(ok.all()),first_token_correct=bool(ok[0]),seconds=entry["median_seconds"])
            result.append(row);groups[(row["condition"],row["group"],row["method"],variant)].append(row)
    summary=[]
    for key,values in groups.items():
        row=dict(zip(["condition","group","method","variant"],key))
        row.update(correct=sum(v["correct"] for v in values),scored=sum(v["scored"] for v in values),exact=sum(v["exact"] for v in values),records=len(values),
          first_token_correct=sum(v["first_token_correct"] for v in values),mean_seconds=sum(v["seconds"] for v in values)/len(values))
        summary.append(row)
    n.write(X/"dev38_grid_score.json",{"rows":result,"summary":summary,"freeze_sha256":n.digest(X/"dev38_grid_freeze.json"),"truth_sha256":n.digest(truthpath),
      "scope":"retrospective exploratory development; no new active canonical method"})
    for row in summary:
        if row["variant"].startswith("step"):print(json.dumps(row),flush=True)
if __name__=="__main__":main()

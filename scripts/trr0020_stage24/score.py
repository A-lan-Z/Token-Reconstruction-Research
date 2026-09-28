"""Validate the complete finite matrix before opening retrospective labels."""
from support import *
import collections,copy
def gate(f):
    expected={(row["id"],cfg[0]) for row in rows() for cfg in CONFIGS}
    cells=[(e["id"],e["method"]) for e in f["entries"]]
    if len(cells)!=96 or len(set(cells))!=96 or set(cells)!=expected:raise ValueError("incomplete matrix")
    if f["binding"]!=binding():raise ValueError("changed sources or assets")
    if len(f["phases"])!=12 or f["warm_anchors"]!=96 or f["mean_anchors"]!=64:raise ValueError("incomplete controls")
    for phase in f["phases"]:
        if n.digest(ROOT/phase["path"])!=phase["sha256"]:raise ValueError("changed phase")
        p=json.loads((ROOT/phase["path"]).read_text())
        if n.digest(ROOT/p["qualification_path"])!=p["qualification_sha256"]:raise ValueError("changed qualification")
        q=json.loads((ROOT/p["qualification_path"]).read_text())
        if not q["passed"] or len(q["runs"])!=6 or len(q["eager"])!=2:raise ValueError("incomplete qualification")
        if {v["length"] for v in q["runs"]}!={40,128}:raise ValueError("missing geometry")
        if not all(v["outputs_and_traces_equal"] and v["stats"]["numeric_valid"] for v in q["runs"]+q["eager"]):raise ValueError("invalid public outputs")
        if len(q["score_reference"])!=2 or not all(v["all_close"] for v in q["score_reference"]):raise ValueError("readout formula")
        if len(q["adjoint_reference"])!=2 or not all(v["identity_passed"] for v in q["adjoint_reference"]):raise ValueError("adjoint formula")
        for v in q["runs"]+q["eager"]+q["adjoint_reference"]:
            if n.digest(ROOT/v["path"])!=v["sha256"]:raise ValueError("changed public output")
    outputs={};mean_count=0
    for entry in f["entries"]:
        data=verify(entry,f["binding"]);warm=int(entry["method"].split("_")[0][4:])
        if not entry["stats"]["numeric_valid"] or entry["stats"]["shortlist_size"] is not None or entry["stats"]["separate_candidate_verification_calls"]!=0:raise ValueError("invalid mechanism")
        anchor=entry["warm_anchor"]
        if not anchor["equal"] or n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise ValueError("warm control failed")
        if warm>0:
            anchor=entry["initial_mean_anchor"];mean_count+=1
            if not anchor["equal"] or n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise ValueError("mean control failed")
        keys={"warm_step"+str(v) for v in [0,16,32,64] if v<=warm}
        keys|={"warm_best_objective","warm_best_observed_error","warm_best_position_soft_error"}
        keys|={f"step{step}_{readout}" for step in CHECKPOINTS for readout in READOUTS}
        if set(data)!=keys|AUX_KEYS:raise ValueError("wrong output keys")
        length=entry["positions"]
        for key,value in data.items():
            if not torch.isfinite(value).all():raise ValueError("nonfinite artifact")
            if key in AUX_KEYS:
                shapes={"initial_embedding":(length-1,2048),"embedding_final":(length,2048),"local_trace":(16,5),
                  "final_position_mse":(length-1,),"final_position_cosine_error":(length-1,),"solver_info":(length,)}
                dtype=torch.int32 if key=="solver_info" else torch.float32
                if tuple(value.shape)!=shapes[key] or value.dtype!=dtype:raise ValueError("invalid diagnostics")
                if key=="solver_info" and (value!=0).any():raise ValueError("failed solve")
            elif tuple(value.shape)!=(length,) or value.dtype!=torch.int64 or int(value[0])!=128000 or (value<0).any() or (value>=128256).any():raise ValueError("invalid tokens")
        outputs[entry["id"],entry["method"]]=data
    if mean_count!=64:raise ValueError("missing means")
    return outputs
def main():
    f=json.loads((X/"dev24_grid_freeze.json").read_text());outputs=gate(f);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(f)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["prefix_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("failed negative truth gate")
    n.write(X/"dev24_grid_truth_gate.json",{"cells":96,"rejected":negative,"freeze_sha256":n.digest(X/"dev24_grid_freeze.json"),"truth_opened_after_unix":time.time()})
    truthpath=INPUT/"evaluator_truth.json";truth=json.loads(truthpath.read_text())
    result=[];groups=collections.defaultdict(list)
    for entry in f["entries"]:
        target=torch.tensor(truth[entry["id"]])
        for variant,ids in outputs[entry["id"],entry["method"]].items():
            if variant in AUX_KEYS:continue
            if ids.shape!=target.shape:raise RuntimeError("length mismatch")
            correct=ids[1:]==target[1:]
            row={key:entry[key] for key in ["id","condition","group","method"]}
            row.update(variant=variant,correct=int(correct.sum()),scored=len(correct),exact=bool(correct.all()))
            result.append(row);groups[(row["condition"],row["group"],row["method"],variant)].append(row)
    summary=[]
    for key,values in groups.items():
        row=dict(zip(["condition","group","method","variant"],key))
        row.update(correct=sum(v["correct"] for v in values),scored=sum(v["scored"] for v in values),exact=sum(v["exact"] for v in values),records=len(values))
        summary.append(row)
    n.write(X/"dev24_grid_score.json",{"rows":result,"summary":summary,"freeze_sha256":n.digest(X/"dev24_grid_freeze.json"),"truth_sha256":n.digest(truthpath),"scope":"retrospective exploratory development; not active canonical method"})
    for row in summary:
        if row["variant"].startswith("step"):print(json.dumps(row),flush=True)
if __name__=="__main__":main()

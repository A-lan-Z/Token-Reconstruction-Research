"""Full-matrix and numerical gates before retrospective truth opening."""
from grid_support import *
import collections,copy
def gate(f):
    expected={(row["id"],cfg[0]) for row in rows() for cfg in CONFIGS}
    cells=[(e["id"],e["method"]) for e in f["entries"]]
    if len(cells)!=56 or len(set(cells))!=56 or set(cells)!=expected:raise ValueError("incomplete matrix")
    if f["binding"]!=binding():raise ValueError("changed sources or assets")
    if len(f["phases"])!=7 or f["warm_anchors"]!=56 or f["mean_anchors"]!=56:raise ValueError("incomplete controls")
    for phase in f["phases"]:
        if n.digest(ROOT/phase["path"])!=phase["sha256"]:raise ValueError("changed phase")
        p=json.loads((ROOT/phase["path"]).read_text())
        if n.digest(ROOT/p["qualification_path"])!=p["qualification_sha256"]:raise ValueError("changed qualification")
        q=json.loads((ROOT/p["qualification_path"]).read_text())
        if not q["passed"] or len(q["runs"])!=6 or len(q["eager"])!=2:raise ValueError("incomplete qualification")
        if {(v["length"],v["repetition"]) for v in q["runs"]}!={(l,r) for l in [40,128] for r in range(3)}:raise ValueError("missing public geometry")
        if not all(v["outputs_and_traces_equal"] and v["stats"]["numeric_valid"] for v in q["runs"]+q["eager"]):raise ValueError("invalid public outputs")
        if len(q["gradient_reference"])!=2 or not all(v["passed"] for v in q["gradient_reference"]):raise ValueError("gradient formula")
        if len(q["one_step_anchors"])!=6 or not all(v["equal"] and n.digest(ROOT/v["path"])==v["sha256"] for v in q["one_step_anchors"]):raise ValueError("one-step anchor failed")
        if len(q["kl_references"])!=2 or not all(len(v["checks"])==4 and all(c["passed"] for c in v["checks"]) for v in q["kl_references"]):raise ValueError("direct KL reference failed")
        for v in q["runs"]+q["eager"]:
            if n.digest(ROOT/v["path"])!=v["sha256"]:raise ValueError("changed public output")
    outputs={}
    for entry in f["entries"]:
        data=verify(entry,f["binding"]);_,factor,iterations=next(v for v in CONFIGS if v[0]==entry["method"]);warm=0
        stats=entry["stats"];length=entry["positions"]
        if stats.get("budget_rule")!="clamp(factor * current observed cosine error,0,1)" or stats["factor"]!=factor or stats["scalar_iterations"]!=iterations:raise ValueError("missing budget rule")
        if not stats["numeric_valid"] or stats["shortlist_size"] is not None or stats["separate_candidate_verification_calls"]!=0:raise ValueError("invalid mechanism")
        if stats["whole_sequence_prefix_forwards"]!=warm+66 or stats["whole_sequence_prefix_backwards"]!=warm+64 or stats["vocabulary_entries_per_sweep"]!=128256 or stats["model_parameter_updates"]!=0:raise ValueError("invalid work accounting")
        for key in ["warm_anchor","initial_mean_anchor"]:
            anchor=entry[key]
            if not anchor["equal"] or n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise ValueError("anchor failed")
        keys={"warm_step"+str(v) for v in [0,16,32,64] if v<=warm}
        keys|={"warm_best_objective","warm_best_observed_error","warm_best_position_soft_error","best_objective","best_position_error"}
        keys|={"step"+str(v) for v in CHECKPOINTS}
        if set(data)!=keys|AUX_KEYS:raise ValueError("wrong output keys")
        shapes={"initial_embedding":(length-1,2048),"mirror_update_trace":(64,5),"final_confidence":(length-1,),"final_position_error":(length-1,)}
        for key,value in data.items():
            if not torch.isfinite(value).all():raise ValueError("nonfinite artifact")
            if key in AUX_KEYS:
                if tuple(value.shape)!=shapes[key] or value.dtype!=torch.float32:raise ValueError("invalid diagnostics")
            elif tuple(value.shape)!=(length,) or value.dtype!=torch.int64 or int(value[0])!=128000 or (value<0).any() or (value>=128256).any():raise ValueError("invalid tokens")
        trace=data["mirror_update_trace"]
        if (trace[:,4]>1.00001).any() or (trace[:,4]<0).any() or (trace[:,0]<0).any() or (trace[:,0]>1.00001).any() or (trace[:,1]>trace[:,0]+2e-5).any() or (trace[:,3]<0).any():raise ValueError("invalid budget update")
        if not torch.equal(data["step0"],data["warm_step"+str(warm)]):raise ValueError("initial token state mismatch")
        outputs[entry["id"],entry["method"]]=data
    return outputs
def main():
    f=json.loads((X/"dev32_grid_freeze.json").read_text());outputs=gate(f);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(f)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["prefix_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("failed negative truth gate")
    n.write(X/"dev32_grid_truth_gate.json",{"cells":56,"rejected":negative,"freeze_sha256":n.digest(X/"dev32_grid_freeze.json"),"truth_opened_after_unix":time.time()})
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
    n.write(X/"dev32_grid_score.json",{"rows":result,"summary":summary,"freeze_sha256":n.digest(X/"dev32_grid_freeze.json"),"truth_sha256":n.digest(truthpath),"scope":"retrospective exploratory development; not active canonical method"})
    for row in summary:
        if row["variant"] in ["step64","best_objective"]:print(json.dumps(row),flush=True)
if __name__=="__main__":main()

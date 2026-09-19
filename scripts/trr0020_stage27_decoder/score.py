"""Reject incomplete or altered evidence before opening retrospective labels."""
from support import *
import collections,copy
def gate(f):
    expected={(row["id"],cfg[0]) for row in rows() for cfg in CONFIGS}
    cells=[(e["id"],e["method"]) for e in f["entries"]]
    if len(cells)!=64 or len(set(cells))!=64 or set(cells)!=expected:raise ValueError("incomplete matrix")
    if f["binding"]!=binding():raise ValueError("changed sources or assets")
    if len(f["phases"])!=8 or f["warm_anchors"]!=64 or f["mean_anchors"]!=64:raise ValueError("incomplete controls")
    for phase in f["phases"]:
        if n.digest(ROOT/phase["path"])!=phase["sha256"]:raise ValueError("changed phase")
        p=json.loads((ROOT/phase["path"]).read_text())
        if n.digest(ROOT/p["qualification_path"])!=p["qualification_sha256"]:raise ValueError("changed qualification")
        q=json.loads((ROOT/p["qualification_path"]).read_text())
        if not q["passed"] or len(q["runs"])!=6:raise ValueError("incomplete qualification")
        if {(v["length"],v["repetition"]) for v in q["runs"]}!={(l,r) for l in [40,128] for r in range(3)}:raise ValueError("missing public geometry")
        if not all(v["outputs_and_traces_equal"] and v["stats"]["numeric_valid"] for v in q["runs"]):raise ValueError("invalid public outputs")
        if len(q["score_reference"])!=2 or not all(v["all_close"] for v in q["score_reference"]):raise ValueError("readout formula")
        if len(q["causal_reference"])!=2 or not all(v["passed"] for v in q["causal_reference"]):raise ValueError("causal context formula")
        for v in q["runs"]:
            if n.digest(ROOT/v["path"])!=v["sha256"]:raise ValueError("changed public output")
    outputs={}
    for entry in f["entries"]:
        data=verify(entry,f["binding"]);cfg=next(v for v in CONFIGS if v[0]==entry["method"]);_,warm,rule,updates=cfg
        stats=entry["stats"];length=entry["positions"];count=(length-1)*updates
        if not stats["numeric_valid"] or stats["shortlist_size"] is not None or stats["separate_candidate_verification_calls"]!=0:raise ValueError("invalid mechanism")
        if (stats["current_forwards"]!=count or stats["current_vjp_calls"]!=count or stats["current_jvp_calls"]!=(count if rule=="cg1" else 0)
            or stats["committed_token_forwards"]!=length-1 or stats["readout_vocabulary_products"]!=2*(length-1)
            or stats["vocabulary_entries_per_sweep"]!=128256 or stats["model_parameter_updates"]!=0):raise ValueError("invalid work accounting")
        for key in ["warm_anchor","initial_mean_anchor"]:
            anchor=entry[key]
            if not anchor["equal"] or n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise ValueError("anchor failed")
        keys={"warm_step"+str(v) for v in [0,16,32,64] if v<=warm}
        keys|={"warm_best_objective","warm_best_observed_error","warm_best_position_soft_error","tokens"}
        if set(data)!=keys|AUX_KEYS:raise ValueError("wrong output keys")
        shapes={"initial_embedding":(length-1,2048),"embedding_final":(length-1,2048),
          "first_prediction":(length-1,2048),"local_trace":(length-1,updates,4),"cache_lengths":(length-1,),"committed_tokens":(length-1,)}
        for key,value in data.items():
            if not torch.isfinite(value).all():raise ValueError("nonfinite artifact")
            if key in AUX_KEYS:
                dtype=torch.int64 if key in ["cache_lengths","committed_tokens"] else torch.float32
                if tuple(value.shape)!=shapes[key] or value.dtype!=dtype:raise ValueError("invalid diagnostics")
            elif tuple(value.shape)!=(length,) or value.dtype!=torch.int64 or int(value[0])!=128000 or (value<0).any() or (value>=128256).any():raise ValueError("invalid tokens")
        if not torch.equal(data["cache_lengths"],torch.arange(1,length)) or not torch.equal(data["committed_tokens"],data["tokens"][:-1]):raise ValueError("causal state mismatch")
        outputs[entry["id"],entry["method"]]=data
    return outputs
def main():
    f=json.loads((X/"dev27_decoder_freeze.json").read_text());outputs=gate(f);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(f)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["prefix_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("failed negative truth gate")
    n.write(X/"dev27_decoder_truth_gate.json",{"cells":64,"rejected":negative,"freeze_sha256":n.digest(X/"dev27_decoder_freeze.json"),"truth_opened_after_unix":time.time()})
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
    n.write(X/"dev27_decoder_score.json",{"rows":result,"summary":summary,"freeze_sha256":n.digest(X/"dev27_decoder_freeze.json"),"truth_sha256":n.digest(truthpath),"scope":"retrospective exploratory development; not active canonical method"})
    for row in summary:
        if row["variant"]=="tokens":print(json.dumps(row),flush=True)
if __name__=="__main__":main()

"""Complete matrix and causal/numerical controls before retrospective truth."""
from support import *
import collections,copy
def gate(f):
    expected={(row["id"],cfg[0]) for row in rows() for cfg in CONFIGS}
    cells=[(e["id"],e["method"]) for e in f["entries"]]
    if len(cells)!=96 or len(set(cells))!=96 or set(cells)!=expected:raise ValueError("incomplete matrix")
    if f["binding"]!=binding() or len(f["phases"])!=3:raise ValueError("changed source or incomplete phases")
    for ref in f["phases"]:
        if n.digest(ROOT/ref["path"])!=ref["sha256"]:raise ValueError("phase changed")
        phase=json.loads((ROOT/ref["path"]).read_text())
        if n.digest(ROOT/phase["qualification_path"])!=phase["qualification_sha256"]:raise ValueError("qualification changed")
        q=json.loads((ROOT/phase["qualification_path"]).read_text())
        if not q["passed"] or len(q["runs"])!=24 or len(q["eager"])!=8 or len(q["history_references"])!=8 or len(q["capture_events"])!=128:raise ValueError("incomplete qualification")
        if not all(v["passed"] for v in q["history_references"]):raise ValueError("history reference")
        expected_qual={(length,cfg[0]) for length in [128,40] for cfg in CONFIGS if cfg[1]==phase["factor"]}
        if {(v["length"],v["method"]) for v in q["eager"]}!=expected_qual:raise ValueError("missing eager case")
        if {(v["length"],v["method"],v["rep"]) for v in q["runs"]}!={(l,m,rep) for l,m in expected_qual for rep in range(3)}:raise ValueError("missing repeats")
        for v in q["runs"]+q["eager"]:
            if not v["exact_repeat"] or not v["stats"]["numeric_valid"] or n.digest(ROOT/v["path"])!=v["sha256"]:raise ValueError("invalid qualification")
    result={}
    for e in f["entries"]:
        data=verify(e,f["binding"]);s=e["stats"];length=e["positions"];_,factor,steps=next(c for c in CONFIGS if c[0]==e["method"])
        if s["factor"]!=factor or s["steps"]!=steps or not s["numeric_valid"] or not s["replay"]:raise ValueError("method mismatch")
        if s["shortlist_size"] is not None or s["separate_candidate_verification_calls"] or s["model_parameter_updates"]:raise ValueError("mechanism mismatch")
        if s["vocabulary"]!=128256 or s["prefix_forwards"]!=(length-1)*(steps+1) or s["prefix_vjps"]!=(length-1)*steps or s["history_commit_forwards"]!=length-1:raise ValueError("work accounting")
        if s["mixture_products"]!=(length-1)*(steps+1) or s["probability_gradient_products"]!=(length-1)*steps or s["initialization_vocabulary_products"]!=length-1:raise ValueError("vocabulary work")
        if s["budget_rule"]!="clamp(factor * current observed cosine error,0,1),four scalar iterations" or not s["last_token_commit_skipped"]:raise ValueError("decision rule")
        if set(data)!={"tokens"}|AUX_KEYS:raise ValueError("wrong arrays")
        ids=data["tokens"]
        if ids.dtype!=torch.long or tuple(ids.shape)!=(length,) or int(ids[0])!=128000 or (ids<0).any() or (ids>=128256).any():raise ValueError("invalid tokens")
        shapes={"soft_error":(length,),"confidence":(length,),"update_trace":(length-1,8,5)}
        for key in AUX_KEYS:
            if data[key].dtype!=torch.float32 or tuple(data[key].shape)!=shapes[key] or not torch.isfinite(data[key]).all():raise ValueError("invalid diagnostics")
        trace=data["update_trace"][:,:steps]
        if (trace[:,:,2]<0).any() or (trace[:,:,2]>1).any() or (trace[:,:,3]>trace[:,:,2]+2e-5).any():raise ValueError("invalid budget trace")
        if torch.count_nonzero(data["update_trace"][:,steps:]):raise ValueError("unused steps not cleared")
        result[e["id"],e["method"]]=ids
    return result
def main():
    freeze=json.loads((X/"dev33_decoder_freeze.json").read_text());outputs=gate(freeze);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(freeze)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["prefix_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("negative gate failed")
    n.write(X/"dev33_decoder_truth_gate.json",{"cells":96,"rejected":negative,"freeze_sha256":n.digest(X/"dev33_decoder_freeze.json"),"truth_opened_after_unix":time.time()})
    truthpath=INPUT/"evaluator_truth.json";truth=json.loads(truthpath.read_text());rows_out=[];groups=collections.defaultdict(list)
    for e in freeze["entries"]:
        target=torch.tensor(truth[e["id"]]);ids=outputs[e["id"],e["method"]]
        if target.shape!=ids.shape:raise RuntimeError("wrong length")
        correct=ids[1:]==target[1:];row={k:e[k] for k in ["id","condition","group","method"]}
        row.update(correct=int(correct.sum()),scored=len(correct),exact=bool(correct.all()))
        rows_out.append(row);groups[row["condition"],row["group"],row["method"]].append(row)
    summary=[]
    for key,values in groups.items():
        row=dict(zip(["condition","group","method"],key));row.update(correct=sum(v["correct"] for v in values),scored=sum(v["scored"] for v in values),exact=sum(v["exact"] for v in values),records=len(values))
        summary.append(row);print(json.dumps(row),flush=True)
    n.write(X/"dev33_decoder_score.json",{"rows":rows_out,"summary":summary,"freeze_sha256":n.digest(X/"dev33_decoder_freeze.json"),"truth_sha256":n.digest(truthpath),"scope":"retrospective eight-record exploratory panel; no canonical method selected"})
if __name__=="__main__":main()

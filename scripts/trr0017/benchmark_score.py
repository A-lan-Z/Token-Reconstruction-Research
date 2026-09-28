from benchmark_support import *
from collections import defaultdict
import copy,numpy as np
from transformers import AutoTokenizer
def main():
    rows=json.loads((INPUT/"metadata.json").read_text())
    entries=[]
    for group in ("controls","continuous","discrete"):
        f=json.loads((X/(group+"_freeze.json")).read_text())
        if f["binding"]!=binding():raise ValueError("binding changed")
        entries+=f["entries"]
    expected={(r["id"],m) for r in rows for m in METHODS}
    actual=[(e["id"],e["method"]) for e in entries]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError("incomplete matrix")
    pp={(e["id"],e["method"]):verify_cell(e) for e in entries}
    for r in rows:
        a=pp[r["id"],"fragment_native"];b=pp[r["id"],"fragment_shared"]
        if not all(torch.equal(a[k],b[k]) for k in a):raise ValueError("shared optimization failed")
    write(X/"truth_gate.json",{"utc":utc(),"verified_cells":len(pp),"truth_read_this_run":False,
         "all_shared_tensors_identical":True,"freeze_hashes":{g:digest(X/(g+"_freeze.json")) for g in ("controls","continuous","discrete")},
         "status":"FULL1360_MATRIX_FROZEN_BEFORE_RETROSPECTIVE_SCORING"})
    truthpath=INPUT/"evaluator_truth.safetensors";truth=n.load_file(str(truthpath))
    tok=AutoTokenizer.from_pretrained(n.ASSETS/"backup",local_files_only=True)
    scored=[];groups=defaultdict(list)
    for e in entries:
        p=pp[e["id"],e["method"]];t=truth[e["id"]];ok=p["tokens"][1:]==t[1:]
        if len(t)!=e["positions"]:raise ValueError("truth geometry changed")
        r={k:e[k] for k in ("id","setup_id","condition","group","method","positions")}
        r.update(correct=int(ok.sum()),scored=len(ok),exact=bool(ok.all()),
                 decoded_exact=tok.decode(p["tokens"][1:].tolist())==tok.decode(t[1:].tolist()),
                 seconds=e["timing"]["total"],candidate_simulations=(len(ok)*256 if e["method"] in METHODS[:3] else 0),
                 continuous_forward_backward=e["timing"].get("continuous_sequence_forward_backward_equivalents",0),
                 discrete_sequence_forward_backward=e["timing"].get("whole_sequence_forward_backward_evaluations",0))
        if "candidates" in p:r["proposal_hits"]=int((p["candidates"][1:]==t[1:,None]).any(-1).sum())
        scored.append(r);groups[(r["setup_id"],r["condition"],r["group"],r["method"])].append(r)
    summary=[]
    for key,rr in groups.items():
        s=dict(zip(("setup_id","condition","group","method"),key))
        s.update(records=len(rr),correct=sum(r["correct"] for r in rr),scored=sum(r["scored"] for r in rr),
            exact=sum(r["exact"] for r in rr),decoded_exact=sum(r["decoded_exact"] for r in rr),
            seconds=sum(r["seconds"] for r in rr),mean_seconds=sum(r["seconds"] for r in rr)/len(rr),
            candidate_simulations=sum(r["candidate_simulations"] for r in rr),
            continuous_forward_backward=sum(r["continuous_forward_backward"] for r in rr),
            discrete_sequence_forward_backward=sum(r["discrete_sequence_forward_backward"] for r in rr))
        s["accuracy"]=s["correct"]/s["scored"]
        if "proposal_hits" in rr[0]:s["proposal_hits"]=sum(r["proposal_hits"] for r in rr)
        summary.append(s)
    paired=[];rng=np.random.default_rng(170017)
    for setup,cond,group in sorted({k[:3] for k in groups}):
        for method,control in (("fragment_shared","fragment_native"),("fragment_shared","a1a2"),
                               ("continuous96","a1a2"),("discrete64","a1a2")):
            aa=sorted(groups[(setup,cond,group,method)],key=lambda x:x["id"])
            bb=sorted(groups[(setup,cond,group,control)],key=lambda x:x["id"])
            assert [r["id"] for r in aa]==[r["id"] for r in bb]
            d=np.array([a["correct"]-b["correct"] for a,b in zip(aa,bb)])
            den=np.array([r["scored"] for r in aa])
            ta=np.array([r["seconds"] for r in aa]);tb=np.array([r["seconds"] for r in bb])
            idx=rng.integers(0,len(aa),(10000,len(aa)))
            paired.append({"setup_id":setup,"condition":cond,"group":group,"method":method,"control":control,
                "accuracy_delta":float(d.sum()/den.sum()),
                "accuracy_delta_bootstrap95":np.quantile(d[idx].sum(1)/den[idx].sum(1),[.025,.975]).tolist(),
                "time_ratio":float(ta.sum()/tb.sum()),
                "time_ratio_bootstrap95":np.quantile(ta[idx].sum(1)/tb[idx].sum(1),[.025,.975]).tolist()})
    result={"task_id":"TRR-0017","summary":summary,"paired":paired,"per_record":scored,
        "truth_sha256":digest(truthpath),"truth_gate_sha256":digest(X/"truth_gate.json"),
        "all_shared_tokens_candidates_scores_mse_byte_identical":True,
        "timing":"one synchronized invocation per method/input; method order rotates for controls; three identical repetitions on maximum geometry qualification; descriptive record bootstrap, not dedicated device noise experiment",
        "scope":"retrospective supplied-public-prefix benchmark; not a recovered-prefix trajectory or fresh confirmation",
        "source_string_recovery":"not defined for token-truncated panels; decoded token exactness reported separately"}
    write(X/"score.json",result)
    oldpath=ROOT/"experiments/TRR-0015/canonical_matrix.json"
    old=json.loads(oldpath.read_text());matrix=copy.deepcopy(old["matrix"])
    for setup in CANONICAL:
        for cell in matrix[setup].values():
            cell["carry_forward_source"]="experiments/TRR-0015/canonical_matrix.json"
            cell["carry_forward_sha256"]=digest(oldpath)
        for r in summary:
            if r["setup_id"]==setup and r["method"] in ("continuous96","discrete64"):
                mid="prefix_parallel_"+r["method"]
                matrix[setup][mid]={"origin":"current retrospective native new method, record batch1, padding stripped",
                       "metrics":r,"source":"experiments/TRR-0017/score.json","source_sha256":digest(X/"score.json")}
    registry=json.loads((X/"registry.json").read_text())
    got={(s,m) for s,cells in matrix.items() for m in cells}
    want={(c["setup_id"],c["method_id"]) for c in registry["required_cells"]}
    if got!=want or len(got)!=56:raise ValueError("incomplete canonical registry")
    write(X/"canonical_matrix.json",{"required_cells":56,"complete":True,"inherited_cells":52,"new_cells":4,
           "matrix":matrix,"current_controls_and_optimized_port":[r for r in summary if r["method"] in METHODS[:3] and r["setup_id"] in CANONICAL],
           "native_control_reproduces_TRR0015_all_tensors":True,"shared_optimization_all_tensors_identical":True,
           "historical_batch8_geometry_caveat":"See TRR-0015 canonical_candidate_geometry_audit.json; current controls preserve its batch1 geometry",
           "inherited_timings_not_compared_to_current":True})
    print(json.dumps({"summary":summary,"paired":paired},indent=2))
if __name__=="__main__":main()


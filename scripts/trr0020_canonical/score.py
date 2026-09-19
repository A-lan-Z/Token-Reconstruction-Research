from support import *
from collections import defaultdict
import copy,numpy as np
def gate(f,rows):
    expected={(r["id"],m,rep) for r in rows for m in METHODS for rep in range(3)}
    actual=[(e["id"],e["method"],e["rep"]) for e in f["entries"]]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError("incomplete replicate matrix")
    if f["binding"]!=binding():raise ValueError("changed source/input binding")
    for phase in f["phases"]:
        if n.digest(ROOT/phase["path"])!=phase["sha256"]:raise ValueError("changed phase receipt")
    outputs={};runs=defaultdict(list)
    for e in f["entries"]:
        if e["binding"]!=f["binding"]:raise ValueError("changed cell binding")
        data=verify(e);key=e["id"],e["method"]
        if key in outputs and not all(torch.equal(v,data[k]) for k,v in outputs[key].items()):raise ValueError("replicate output change")
        outputs[key]=data;runs[key].append(e)
    return outputs,runs
def main():
    torch.set_num_threads(2);rows=json.loads((X/"metadata.json").read_text())
    f=json.loads((X/"prediction_freeze.json").read_text());outputs,runs=gate(f,rows);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(f)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["prefix_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad,rows)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("truth-gate negative tests failed")
    n.write(X/"truth_gate.json",{"scope":"full retrospective matrix frozen before current labels","replicate_cells":len(f["entries"]),"unique_outputs":len(outputs),
      "negative_cases_rejected":negative,"freeze_sha256":n.digest(X/"prediction_freeze.json"),"utc":utc()})
    truthpath=INPUT/"evaluator_truth.safetensors";truth=n.load_file(str(truthpath))
    per=[];groups=defaultdict(list)
    for row in rows:
        target=truth[row["id"]];assert len(target)==row["positions"]
        for method in METHODS:
            p=outputs[row["id"],method];entries=runs[row["id"],method]
            e=sorted(entries,key=lambda e:e["phases"]["total"])[1];ok=p["tokens"][1:]==target[1:]
            r={**row,"method":method,"correct":int(ok.sum()),"scored":len(ok),"exact":bool(ok.all()),"seconds":e["phases"]["total"],
               "phase_seconds":e["phases"],"prefix_forward_count":e["stats"].get("whole_sequence_prefix_forwards"),
               "prefix_backward_count":e["stats"].get("whole_sequence_prefix_backwards"),"candidate_simulations":e["stats"].get("candidate_simulations",0)}
            if method not in METHOD_IDS:
                hits=(p["candidates"][1:]==target[1:,None]).any(-1);r.update(proposal_hits=int(hits.sum()),selector_errors=int((hits&~ok).sum()))
            else:r.update(proposal_hits=None,selector_errors=None)
            per.append(r);groups[row["setup_id"],method].append(r)
    summary=[]
    for (setup,method),items in groups.items():
        correct=sum(r["correct"] for r in items);scored=sum(r["scored"] for r in items)
        summary.append({"setup_id":setup,"method":method,"records":len(items),"correct":correct,"scored":scored,"accuracy":correct/scored,
         "exact":sum(r["exact"] for r in items),"seconds_per_record":float(np.mean([r["seconds"] for r in items])),
         "candidate_simulations":sum(r["candidate_simulations"] for r in items),
         "prefix_forward_count":sum(r["prefix_forward_count"] or 0 for r in items),"prefix_backward_count":sum(r["prefix_backward_count"] or 0 for r in items)})
    for setup,count in [(CANONICAL[0],2490),(CANONICAL[1],13952)]:
        for method in ["a1_native","a1_graph"]:
            if next(r for r in summary if r["setup_id"]==setup and r["method"]==method)["correct"]!=count:raise RuntimeError("baseline accuracy anchor failed")
    paired=[];rng=np.random.default_rng(200032)
    for setup in CANONICAL:
        for method in METHOD_IDS:
            for control in ["a1_native","a1_graph"]:
                a=sorted(groups[setup,method],key=lambda r:r["id"]);b=sorted(groups[setup,control],key=lambda r:r["id"])
                assert [r["id"] for r in a]==[r["id"] for r in b]
                at=np.array([r["seconds"] for r in a]);bt=np.array([r["seconds"] for r in b])
                delta=np.array([x["correct"]-y["correct"] for x,y in zip(a,b)]);den=np.array([r["scored"] for r in a])
                ix=rng.integers(0,len(a),size=(10000,len(a)))
                paired.append({"setup_id":setup,"method":method,"control":control,"correct_delta":int(delta.sum()),
                  "accuracy_delta_interval95":np.quantile(delta[ix].sum(-1)/den[ix].sum(-1),[.025,.975]).tolist(),
                  "runtime_ratio":float(at.sum()/bt.sum()),"runtime_ratio_interval95":np.quantile(at[ix].sum(-1)/bt[ix].sum(-1),[.025,.975]).tolist(),
                  "exact_delta":sum(x["exact"]-y["exact"] for x,y in zip(a,b))})
    score={"task_id":"TRR-0020","scope":"retrospective canonical comparison; public supplied prefix, not recovered-prefix trajectory",
      "summary":summary,"paired":paired,"per_record":per,"truth_sha256":n.digest(truthpath),"freeze_sha256":n.digest(X/"prediction_freeze.json"),"finished_utc":utc()}
    n.write(X/"score.json",score)
    inherited=json.loads((X/"inherited_matrix.json").read_text());matrix=copy.deepcopy(inherited["matrix"])
    reg=json.loads((X/"registry.json").read_text())
    for setup,cells in matrix.items():
        for cell in cells.values():
            cell["carried_forward_via"]={"path":str((X/"inherited_matrix.json").relative_to(ROOT)),"sha256":n.digest(X/"inherited_matrix.json")}
            cell["timing_comparable_to_current_run"]=False
        for name,method_id in METHOD_IDS.items():
            cells[method_id]={"origin":"current whole-sequence full-vocabulary method; retrospective","source":str((X/"score.json").relative_to(ROOT)),
              "source_sha256":n.digest(X/"score.json"),"metrics":next(r for r in summary if r["setup_id"]==setup and r["method"]==name),
              "decision_rule":reg["decision_rules"][name],"port_differences":[]}
    expected={(r["setup_id"],r["method_id"]) for r in reg["required_cells"]}
    if {(setup,m) for setup,cells in matrix.items() for m in cells}!=expected:raise RuntimeError("canonical matrix incomplete")
    n.write(X/"canonical_matrix.json",{"required_cells":66,"complete":True,"inherited_cells":62,"new_cells":4,"matrix":matrix,
      "source_registry_sha256":n.digest(X/"registry.json"),"current_controls":summary,"inherited_timings_not_compared_to_current":True})
    print(json.dumps({"summary":summary,"paired":paired},indent=2),flush=True)
if __name__=="__main__":main()

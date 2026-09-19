"""Validate the whole frozen matrix before opening retrospective evaluator truth."""
from support import *
import copy
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer

def verify_matrix(receipt,rows):
    expected={(r["id"],m) for r in rows for m in METHODS}
    actual=[(r["id"],r["method"]) for r in receipt["entries"]]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError("incomplete or duplicate matrix")
    check_binding(receipt["binding"])
    byid={r["id"]:r for r in rows};predictions={}
    for entry in receipt["entries"]:
        if entry["binding"] != receipt["binding"]:raise ValueError("cell binding changed")
        if entry["setup_id"] != byid[entry["id"]]["setup_id"]:raise ValueError("setup changed")
        predictions[(entry["id"],entry["method"])]=verify_cell(entry,byid[entry["id"]])
    for row in rows:
        a=predictions[(row["id"],"fragment256")]["candidates"]
        b=predictions[(row["id"],"fragment512")]["candidates"]
        if not torch.equal(a,b[:,:256]):raise ValueError("subset changed")
    return predictions

def evaluator_truth(rows):
    # Called only after verification and the truth-opening gate are written.
    truth={}
    old=json.loads((PREVIOUS/"outputs/TRR-0014/fresh_r2/evaluator_truth.json").read_text())
    sys.path.insert(0,str(ROOT/"scripts"))
    import trr0001_r2_dual_benchmark as dual
    clean_path=MAIN/"outputs/TRR-0001-R1/clean/evaluator_private/blind_truth.jsonl"
    clean,clean_ids=dual.load_new_truth(clean_path,64)
    historical_root=MAIN.parent/"backdoor_lora/ersoy2026"
    source_path=historical_root/"scripts/score_a1_a2_source300_20260809.py"
    loader=dual.import_path("trr0015_evaluator_source300",source_path)
    config,captures,h,mask,pos=dual.historical_inputs(historical_root,loader)
    finance,finance_ids=dual.load_old_truth(loader,captures,config)
    for row in rows:
        length=row["positions"]
        if row["setup_id"]=="original_r2":
            values=old[row["original_id"]]
        elif row["setup_id"]==CANONICAL[0]:
            values=clean[row["source_row"]].tolist()
        else:
            index=row["source_row"]
            assert int(mask[index].sum())==length
            values=finance[index,:length].tolist()
        assert len(values)==length and values[0]==128000
        truth[row["id"]]=torch.tensor(values)
    hashes={str(p):digest(p) for p in [
        PREVIOUS/"outputs/TRR-0014/fresh_r2/evaluator_truth.json",clean_path,source_path,
        historical_root/"config/a1_a2_source300_static_20260809.json",
        loader.resolve_inside_ersoy(config["source"]["path"])]}
    return truth,hashes

def inherited_matrix(summary):
    matrix={s:{} for s in CANONICAL}
    sources=[
        ("experiments/TRR-0002/crossover/result.json",None),
        ("experiments/TRR-0002/calibrated-dual/result.json","a1_scale_calibrated_adaptive_causal_k32_to64"),
        ("experiments/TRR-0002/configuration-search/canonical/result.json","a1_a2_exhaustive_configuration_winner"),
    ]
    for rel,method in sources:
        data=json.loads((ROOT/rel).read_text());sha=digest(ROOT/rel)
        for setup in CANONICAL:
            cells=data["matrix"][setup] if method is None else {method:data["setups"][setup]}
            for method_id,cell in cells.items():
                matrix[setup][method_id]={
                    "origin":"inherited previously completed frozen result; not rerun",
                    "source":rel,"source_sha256":sha,"metrics":cell["metrics"],
                    "cost":cell.get("cost",cell.get("timing")),
                    "port_differences":cell.get("port_differences"),
                    "timing_comparable_to_current_run":False,
                }
    for row in summary:
        if row["setup_id"] not in CANONICAL or row["method"]=="a1a2":continue
        method_id="prefix_native_"+row["method"]
        matrix[row["setup_id"]][method_id]={
            "origin":"current retrospective benchmark-compatible port",
            "source":"experiments/TRR-0015/score.json","metrics":row,
            "port_differences":["record_batch1","trim only right padding","input/output serialization"],
            "decision_rule":"unchanged native direct cosine; original candidate order; fixed256 or512",
        }
    registry=json.loads((X/"registry.json").read_text())
    expected={(c["setup_id"],c["method_id"]) for c in registry["required_cells"]}
    actual={(s,m) for s,cells in matrix.items() for m in cells}
    if actual!=expected or len(actual)!=52:raise ValueError("incomplete canonical registry matrix")
    return matrix

def main():
    torch.set_num_threads(2)
    rows=json.loads((OUT/"metadata.json").read_text())
    receipt_path=X/"prediction_receipt.json";receipt=json.loads(receipt_path.read_text())
    predictions=verify_matrix(receipt,rows)
    negative=[]
    broken=copy.deepcopy(receipt);broken["entries"]=broken["entries"][:-1]
    try:verify_matrix(broken,rows)
    except ValueError:negative.append("missing cell")
    broken=copy.deepcopy(receipt);broken["entries"][0]["sha256"]="0"*64
    try:verify_matrix(broken,rows)
    except ValueError:negative.append("changed output")
    broken=copy.deepcopy(receipt);key=next(iter(broken["binding"]["implementation_hashes"]))
    broken["binding"]["implementation_hashes"][key]="0"*64
    try:verify_matrix(broken,rows)
    except ValueError:negative.append("changed implementation")
    if len(negative)!=3:raise RuntimeError("negative gate failure")
    write(X/"truth_gate.json",{"utc":utc(),"verified_cells":len(predictions),"negative_cases_rejected":negative,
                              "prediction_receipt_sha256":digest(receipt_path),"truth_read_this_run":False,
                              "status":"RETROSPECTIVE_OUTPUTS_FROZEN_BEFORE_SCORER_LABEL_ACCESS"})
    truth,truth_hashes=evaluator_truth(rows)
    tokenizer=AutoTokenizer.from_pretrained(n.ASSETS/"backup",local_files_only=True)
    entries={(e["id"],e["method"]):e for e in receipt["entries"]}
    scored=[]
    for row in rows:
        t=truth[row["id"]]
        decoded_truth=tokenizer.decode(t[1:].tolist(),skip_special_tokens=False)
        for method in METHODS:
            p=predictions[(row["id"],method)];ok=p["tokens"][1:]==t[1:]
            hits=(p["candidates"][1:]==t[1:,None]).any(-1)
            entry=entries[(row["id"],method)]
            median_phase=sorted(entry["phases"],key=lambda p:p["total"])[1]
            scored.append({**row,"method":method,"correct":int(ok.sum()),"scored":len(t)-1,
                           "exact":bool(ok.all()),"decoded_exact":tokenizer.decode(p["tokens"][1:].tolist(),skip_special_tokens=False)==decoded_truth,
                           "proposal_hits":int(hits.sum()),"wrong_despite_inclusion":int((~ok&hits).sum()),
                           "first_error":next((i+1 for i,v in enumerate(ok) if not bool(v)),None),
                           "seconds":entry["median_seconds"],"phases":median_phase,
                           "logical_simulations":entry["logical_simulations"]})
    groups=defaultdict(list)
    for row in scored:groups[(row["setup_id"],row["condition"],row["group"],row["method"])].append(row)
    summary=[]
    for (setup,condition,group,method),rr in groups.items():
        total=sum(r["scored"] for r in rr);correct=sum(r["correct"] for r in rr);hits=sum(r["proposal_hits"] for r in rr)
        summary.append({"setup_id":setup,"condition":condition,"group":group,"method":method,"records":len(rr),
                        "correct":correct,"scored":total,"accuracy":correct/total,"exact":sum(r["exact"] for r in rr),
                        "decoded_exact":sum(r["decoded_exact"] for r in rr),"proposal_hits":hits,"proposal_recall":hits/total,
                        "conditional_selector_accuracy":correct/hits if hits else None,
                        "wrong_despite_inclusion":sum(r["wrong_despite_inclusion"] for r in rr),
                        "seconds":sum(r["seconds"] for r in rr),"mean_record_seconds":sum(r["seconds"] for r in rr)/len(rr),
                        "phase_seconds":{k:sum(r["phases"][k] for r in rr) for k in rr[0]["phases"]},
                        "logical_simulations":sum(r["logical_simulations"] for r in rr)})
    paired=[];rng=np.random.default_rng(1515256)
    for setup,condition,group in sorted({k[:3] for k in groups}):
        a=groups[(setup,condition,group,"fragment256")]
        for comparator in ("a1a2","fragment512"):
            b=groups[(setup,condition,group,comparator)]
            assert [r["id"] for r in a]==[r["id"] for r in b]
            denominators=np.array([r["scored"] for r in a])
            delta=np.array([x["correct"]-y["correct"] for x,y in zip(a,b)])
            exact=np.array([int(x["exact"])-int(y["exact"]) for x,y in zip(a,b)])
            indices=rng.integers(0,len(a),(10000,len(a)))
            token_draws=delta[indices].sum(1)/denominators[indices].sum(1)
            paired.append({"setup_id":setup,"condition":condition,"group":group,"method":"fragment256","comparator":comparator,
                           "correct_token_delta":int(delta.sum()),"token_accuracy_delta":float(delta.sum()/denominators.sum()),
                           "token_accuracy_bootstrap95":np.quantile(token_draws,[.025,.975]).tolist(),
                           "exact_record_delta":int(exact.sum()),"exact_rate_bootstrap95":np.quantile(exact[indices].mean(1),[.025,.975]).tolist(),
                           "better_records":int((delta>0).sum()),"tied_records":int((delta==0).sum()),"worse_records":int((delta<0).sum()),
                           "time_ratio":sum(r["seconds"] for r in a)/sum(r["seconds"] for r in b)})
    result={"task_id":"TRR-0015","status":"RETROSPECTIVE_EQUAL_BUDGET_MATRIX_SCORED",
            "summary":summary,"paired":paired,"per_record":scored,"truth_hashes":truth_hashes,
            "truth_gate_sha256":digest(X/"truth_gate.json"),"prediction_receipt_sha256":digest(receipt_path),
            "source_string_recovery":"unavailable/not defined for token-truncated source panels; not equated to decoded-token exactness",
            "actual_recovered_prefix_used":False,"method_changed_after_scores":False,
            "uncertainty":"paired record bootstrap; conditional on these already-opened panels; no equivalence test",
            "finished_utc":utc()}
    matrix=inherited_matrix(summary)
    write(X/"score.json",result)
    write(X/"canonical_matrix.json",{"required_cells":52,"complete":True,"inherited_cells":48,"new_cells":4,
                                    "matrix":matrix,"current_baseline_ports":[r for r in summary if r["setup_id"] in CANONICAL and r["method"]=="a1a2"],
                                    "all_current_ports_share_timing_boundary_and_hardware":True,
                                    "inherited_timings_not_compared_to_current":True})
    print(json.dumps({"summary":summary,"paired":paired},indent=2),flush=True)
if __name__=="__main__":main()

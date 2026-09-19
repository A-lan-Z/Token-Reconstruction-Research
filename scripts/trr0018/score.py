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
    return predictions

def evaluator_truth(rows):
    previous=ROOT.parent/"TRR-0016/outputs/TRR-0016/evaluator_truth.safetensors"
    old=load_file(str(previous));fresh_path=OUT/"fresh_r4/evaluator_truth.json"
    fresh=json.loads(fresh_path.read_text());truth={}
    for row in rows:
        values=torch.tensor(fresh[row['original_id']]) if row['setup_id']=='fresh_r4' else old[row['id']]
        assert len(values)==row['positions'] and int(values[0])==128000
        truth[row['id']]=values
    save_file(truth,str(OUT/'evaluator_truth.safetensors'))
    return truth,{str(previous):digest(previous),str(fresh_path):digest(fresh_path)}

def inherited_matrix(summary):
    sources=[ROOT/'experiments/TRR-0016/canonical_matrix.json',X/'inherited/TRR-0017/canonical_matrix.json']
    matrix={s:{} for s in CANONICAL}
    for path in sources:
        previous=json.loads(path.read_text())['matrix']
        for setup,cells in previous.items():
            for method,entry in cells.items():
                if method in matrix[setup]:continue
                cell=copy.deepcopy(entry);cell['carried_forward_via']={'path':str(path.relative_to(ROOT)),'sha256':digest(path)}
                cell['timing_comparable_to_current_run']=False;matrix[setup][method]=cell
    assert sum(map(len,matrix.values()))==58
    for row in summary:
        if row['setup_id'] not in CANONICAL or row['method']!='mixed256':continue
        matrix[row['setup_id']]['prefix_native_fragment_mixed256']={
            'origin':'current retrospective benchmark-compatible port','source':'experiments/TRR-0018/score.json','metrics':row,
            'port_differences':['record_batch1','trim only right padding','input/output serialization'],
            'decision_rule':'base64+64 preserved;128+128 suffix parents;64 vote slots then existing lookup similarity;fixed256 native direct cosine'}
    registry=json.loads((X/'registry.json').read_text())
    expected={(c['setup_id'],c['method_id']) for c in registry['required_cells']}
    actual={(s,m) for s,cells in matrix.items() for m in cells}
    if actual!=expected or len(actual)!=60:raise ValueError('incomplete canonical matrix')
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
                              "status":"FULL_MATRIX_FROZEN_BEFORE_SCORER_LABEL_ACCESS; FRESH_R4_PREVIOUSLY_UNOPENED"})
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
    paired=[];rng=np.random.default_rng(1818256)
    for setup,condition,group in sorted({k[:3] for k in groups}):
        a=groups[(setup,condition,group,"mixed256")]
        for comparator in ("a1a2","frequency256"):
            b=groups[(setup,condition,group,comparator)]
            assert [r["id"] for r in a]==[r["id"] for r in b]
            denominators=np.array([r["scored"] for r in a])
            delta=np.array([x["correct"]-y["correct"] for x,y in zip(a,b)])
            exact=np.array([int(x["exact"])-int(y["exact"]) for x,y in zip(a,b)])
            indices=rng.integers(0,len(a),(10000,len(a)))
            token_draws=delta[indices].sum(1)/denominators[indices].sum(1)
            paired.append({"setup_id":setup,"condition":condition,"group":group,"method":"mixed256","comparator":comparator,
                           "correct_token_delta":int(delta.sum()),"token_accuracy_delta":float(delta.sum()/denominators.sum()),
                           "token_accuracy_bootstrap95":np.quantile(token_draws,[.025,.975]).tolist(),
                           "exact_record_delta":int(exact.sum()),"exact_rate_bootstrap95":np.quantile(exact[indices].mean(1),[.025,.975]).tolist(),
                           "better_records":int((delta>0).sum()),"tied_records":int((delta==0).sum()),"worse_records":int((delta<0).sum()),
                           "time_ratio":sum(r["seconds"] for r in a)/sum(r["seconds"] for r in b)})
    result={"task_id":"TRR-0018","status":"CANONICAL_R2_R3_RETROSPECTIVE_PLUS_FRESH_R4_CONFIRMATION",
            "summary":summary,"paired":paired,"per_record":scored,"truth_hashes":truth_hashes,
            "truth_gate_sha256":digest(X/"truth_gate.json"),"prediction_receipt_sha256":digest(receipt_path),
            "source_string_recovery":"unavailable/not defined for token-truncated source panels; not equated to decoded-token exactness",
            "actual_recovered_prefix_used":False,"method_changed_after_scores":False,
            "uncertainty":"paired record bootstrap; canonical, R2 and R3 retrospective; R4 fresh; no equivalence test",
            "finished_utc":utc()}
    matrix=inherited_matrix(summary)
    write(X/"score.json",result)
    write(X/"canonical_matrix.json",{"required_cells":60,"complete":True,"inherited_cells":58,"new_cells":2,
                                    "matrix":matrix,"current_baseline_ports":[r for r in summary if r["setup_id"] in CANONICAL and r["method"]=="a1a2"],
                                    "all_current_ports_share_timing_boundary_and_hardware":True,
                                    "inherited_timings_not_compared_to_current":True})
    print(json.dumps({"summary":summary,"paired":paired},indent=2),flush=True)
if __name__=="__main__":main()

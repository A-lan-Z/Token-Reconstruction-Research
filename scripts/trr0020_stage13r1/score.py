"""Open retrospective labels only after validating the complete fixed matrix."""
from pathlib import Path
import sys,json,collections,copy,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
import native as n
X=ROOT/"experiments/TRR-0020"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"

def gate(f):
    names={f"lambda{str(strength).replace('.','p')}_lr{str(lr).replace('.','p')}" for strength in [0.,.003,.01,.03] for lr in [.001,.003,.01]}
    selected=[];counts={}
    for r in json.loads((INPUT/"metadata.json").read_text()):
        key=(r["condition"],r["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(r);counts[key]+=1
    positions={r["id"]:r["positions"] for r in selected}
    expected={(r["id"],name) for r in selected for name in names}
    cells=[(e["id"],e["method"]) for e in f["entries"]]
    if len(cells)!=96 or len(set(cells))!=96 or set(cells)!=expected:raise ValueError("incomplete matrix")
    for path,sha in f["binding"]["sources"].items():
        if n.digest(ROOT/path)!=sha:raise ValueError("changed bound source")
    for key,path in [("observations_sha256",INPUT/"observations.safetensors"),
      ("metadata_sha256",INPUT/"metadata.json"),("prefix_sha256",n.ASSETS/"backup/prefix.safetensors")]:
        if n.digest(path)!=f["binding"][key]:raise ValueError("changed bound asset")
    if f.get("original_cell_anchors")!=88 or f.get("public_config_anchors")!=12:
        raise ValueError("incomplete isolation anchors")
    for phase in f["phases"]:
        if n.digest(ROOT/phase["path"])!=phase["sha256"]:raise ValueError("changed phase")
        q=json.loads((ROOT/phase["path"]).read_text())["qualification"]
        if not all(r["equal_to_original"] for r in q["runs"]):raise ValueError("failed public equivalence")
        for r in q["runs"]:
            if n.digest(ROOT/r["path"])!=r["sha256"]:raise ValueError("changed public qualification")
    outputs={};anchor_count=0
    for e in f["entries"]:
        if e["binding"]!=f["binding"]:raise ValueError("changed cell binding")
        if n.digest(ROOT/e["path"])!=e["sha256"]:raise ValueError("changed prediction")
        if e["stats"]["shortlist_size"] is not None or e["stats"]["separate_candidate_verification_calls"]!=0:
            raise ValueError("forbidden candidate mechanism")
        anchor=e.get("original_anchor")
        if anchor is not None:
            anchor_count+=1
            if not anchor["all_outputs_and_traces_equal"] or n.digest(ROOT/anchor["path"])!=anchor["sha256"]:
                raise ValueError("changed original equivalence anchor")
        data=n.load_file(str(ROOT/e["path"]))
        expected_keys={"step0","step16","step32","step64","step128","best_objective","best_position_error"}
        if set(data)!=expected_keys:raise ValueError("wrong outputs")
        for ids in data.values():
            if tuple(ids.shape)!=(positions[e["id"]],) or ids.dtype!=n.torch.int64 or int(ids[0])!=128000:
                raise ValueError("invalid tokens/BOS")
            if (ids<0).any() or (ids>=128256).any():raise ValueError("out of vocabulary")
        outputs[e["id"],e["method"]]=data
    if anchor_count!=88:raise ValueError("missing original cell anchors")
    return outputs

def main():
    f=json.loads((X/"dev13_r1_freeze.json").read_text());outputs=gate(f);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(f)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["prefix_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("negative truth gate failed")
    n.write(X/"dev13_r1_truth_gate.json",{"cells":96,"rejected":negative,"freeze_sha256":n.digest(X/"dev13_r1_freeze.json"),
      "truth_opened_after_unix":time.time()})
    truthpath=INPUT/"evaluator_truth.json";truth=json.loads(truthpath.read_text())
    rows=[];groups=collections.defaultdict(list)
    for e in f["entries"]:
        target=n.torch.tensor(truth[e["id"]])
        for variant,ids in outputs[e["id"],e["method"]].items():
            if ids.shape!=target.shape:raise RuntimeError("length mismatch")
            correct=ids[1:]==target[1:]
            row={k:e[k] for k in ("id","condition","group","method")}
            row.update(variant=variant,correct=int(correct.sum()),scored=len(correct),exact=bool(correct.all()))
            rows.append(row);groups[(row["condition"],row["group"],row["method"],variant)].append(row)
    summary=[]
    for key,values in groups.items():
        row=dict(zip(("condition","group","method","variant"),key))
        row.update(correct=sum(v["correct"] for v in values),scored=sum(v["scored"] for v in values),
          exact=sum(v["exact"] for v in values),records=len(values))
        summary.append(row)
    n.write(X/"dev13_r1_score.json",{"rows":rows,"summary":summary,"freeze_sha256":n.digest(X/"dev13_r1_freeze.json"),
      "truth_sha256":n.digest(truthpath),"scope":"retrospective exploratory development; not active canonical method"})
    for row in summary:
        if row["variant"] in ("step128","best_objective","best_position_error"):print(json.dumps(row),flush=True)
if __name__=="__main__":main()

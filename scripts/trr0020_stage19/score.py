"""Open retrospective labels only after validating the complete fixed matrix."""
from pathlib import Path
import sys,json,collections,copy,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
import native as n
X=ROOT/"experiments/TRR-0020"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"

def gate(f):
    names={'warm'+str(v) for v in [32,64,128,256]}
    selected=[];counts={}
    for r in json.loads((INPUT/"metadata.json").read_text()):
        key=(r["condition"],r["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(r);counts[key]+=1
    positions={r["id"]:r["positions"] for r in selected}
    expected={(r["id"],name) for r in selected for name in names}
    cells=[(e["id"],e["method"]) for e in f["entries"]]
    if len(cells)!=32 or len(set(cells))!=32 or set(cells)!=expected:raise ValueError("incomplete matrix")
    for path,sha in f["binding"]["sources"].items():
        if n.digest(ROOT/path)!=sha:raise ValueError("changed bound source")
    for key,path in [("observations_sha256",INPUT/"observations.safetensors"),
      ("metadata_sha256",INPUT/"metadata.json"),("prefix_sha256",n.ASSETS/"backup/prefix.safetensors"),("config_sha256",n.ASSETS/"backup/config.json")]:
        if n.digest(path)!=f["binding"][key]:raise ValueError("changed bound asset")
    if f.get("control_anchors")!=32:
        raise ValueError("incomplete isolation anchors")
    if len(f["phases"])!=4:raise ValueError("incomplete phases")
    for phase in f["phases"]:
        if n.digest(ROOT/phase["path"])!=phase["sha256"]:raise ValueError("changed phase")
        q=json.loads((ROOT/phase["path"]).read_text())["qualification"]
        if not all(r["stats"]["readout_numeric_valid"] for r in q["runs"]):raise ValueError("invalid public score")
        if len(q["score_reference"])!=2 or not all(r["all_close"] for r in q["score_reference"]):raise ValueError("failed score formula")
        if not all(r["outputs_and_traces_equal"] for r in q["runs"]):raise ValueError("failed public equivalence")
        if {r["length"] for r in q["runs"]}!={40,128}:raise ValueError("missing public geometry")
        if len(q["controls"])!=2 or not all(r["tokens_equal"] and r["traces_equal"] for r in q["controls"]):
            raise ValueError("failed eager reference")
        for r in q["runs"]+q["controls"]:
            if n.digest(ROOT/r["path"])!=r["sha256"]:raise ValueError("changed public qualification")
    outputs={};anchor_count=0
    for e in f["entries"]:
        if e["binding"]!=f["binding"]:raise ValueError("changed cell binding")
        if n.digest(ROOT/e["path"])!=e["sha256"]:raise ValueError("changed prediction")
        if e["stats"]["shortlist_size"] is not None or e["stats"]["separate_candidate_verification_calls"]!=0:
            raise ValueError("forbidden candidate mechanism")
        if not e["stats"]["readout_numeric_valid"]:raise ValueError("invalid score")
        anchor=e.get("original_anchor")
        if anchor is not None:
            anchor_count+=1
            if not anchor["snapshots_and_trace_prefixes_equal"] or n.digest(ROOT/anchor["path"])!=anchor["sha256"]:
                raise ValueError("changed original equivalence anchor")
        data=n.load_file(str(ROOT/e["path"]))
        warm=int(e["method"].split("_")[0][4:])
        expected_keys={'raw_l2','raw_cosine','white_l2','white_cosine','fitted_embedding','final_confidence'}
        expected_keys|={"warm_step"+str(i) for i in [0,16,32,64,128,256] if i<=warm}
        expected_keys|={"warm_best_objective","warm_best_observed_error","warm_best_position_soft_error"}
        if set(data)!=expected_keys:raise ValueError("wrong outputs")
        for name,ids in data.items():
            if name in ['fitted_embedding','final_confidence']:
                shape=(positions[e['id']]-1,2048) if name=='fitted_embedding' else (positions[e['id']]-1,)
                if tuple(ids.shape)!=shape or ids.dtype!=n.torch.float32 or not n.torch.isfinite(ids).all():raise ValueError('invalid fitted embedding/confidence')
                continue
            if tuple(ids.shape)!=(positions[e["id"]],) or ids.dtype!=n.torch.int64 or int(ids[0])!=128000:
                raise ValueError("invalid tokens/BOS")
            if (ids<0).any() or (ids>=128256).any():raise ValueError("out of vocabulary")
        outputs[e["id"],e["method"]]=data
    if anchor_count!=32:raise ValueError("missing original cell anchors")
    return outputs

def main():
    f=json.loads((X/"dev19_freeze.json").read_text());outputs=gate(f);negative=[]
    for mode in ["missing_cell","source_changed","prediction_changed"]:
        bad=copy.deepcopy(f)
        if mode=="missing_cell":bad["entries"].pop()
        elif mode=="source_changed":bad["binding"]["prefix_sha256"]="0"*64
        else:bad["entries"][0]["sha256"]="0"*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError("negative truth gate failed")
    n.write(X/"dev19_truth_gate.json",{"cells":32,"rejected":negative,"freeze_sha256":n.digest(X/"dev19_freeze.json"),
      "truth_opened_after_unix":time.time()})
    truthpath=INPUT/"evaluator_truth.json";truth=json.loads(truthpath.read_text())
    rows=[];groups=collections.defaultdict(list)
    for e in f["entries"]:
        target=n.torch.tensor(truth[e["id"]])
        for variant,ids in outputs[e["id"],e["method"]].items():
            if variant in ["fitted_embedding","final_confidence"]:continue
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
    n.write(X/"dev19_score.json",{"rows":rows,"summary":summary,"freeze_sha256":n.digest(X/"dev19_freeze.json"),
      "truth_sha256":n.digest(truthpath),"scope":"retrospective exploratory development; not active canonical method"})
    for row in summary:
        if row["variant"] in ("raw_l2","raw_cosine","white_l2","white_cosine","warm_step"+row["method"][4:]):print(json.dumps(row),flush=True)
if __name__=="__main__":main()

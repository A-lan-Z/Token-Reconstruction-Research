"""Verify the whole development freeze BEFORE opening retrospective labels."""
from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
import native as n
X=ROOT/"experiments/TRR-0020"
f=json.loads((X/"dev11_freeze.json").read_text())
if len(f["entries"])!=96:raise RuntimeError("incomplete matrix")
for path,sha in f["binding"]["sources"].items():
    if n.digest(ROOT/path)!=sha:raise RuntimeError("changed source "+path)
for e in f["entries"]:
    if n.digest(ROOT/e["path"])!=e["sha256"]:raise RuntimeError("changed prediction")
    if e["stats"]["shortlist_size"] is not None or e["stats"]["separate_candidate_verification_calls"]!=0:
        raise RuntimeError("candidate shortlist is forbidden")
truthpath=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1/evaluator_truth.json"
truth=json.loads(truthpath.read_text());rows=[];groups=collections.defaultdict(list)
for e in f["entries"]:
    predicted=n.load_file(str(ROOT/e["path"]));target=n.torch.tensor(truth[e["id"]])
    for variant,ids in predicted.items():
        correct=ids[1:]==target[1:]
        r={k:e[k] for k in ("id","condition","group","method")}
        r.update(variant=variant,correct=int(correct.sum()),scored=len(correct),exact=bool(correct.all()))
        rows.append(r);groups[(r["condition"],r["group"],r["method"],variant)].append(r)
summary=[]
for key,values in groups.items():
    row=dict(zip(("condition","group","method","variant"),key))
    row.update(correct=sum(v["correct"] for v in values),scored=sum(v["scored"] for v in values),
               exact=sum(v["exact"] for v in values),records=len(values))
    summary.append(row)
n.write(X/"dev11_score.json",{"rows":rows,"summary":summary,
       "freeze_sha256":n.digest(X/"dev11_freeze.json"),"truth_sha256":n.digest(truthpath),
       "scope":"retrospective exploratory development; not active dual-benchmark comparison"})
for row in summary:
    if row["variant"] in ("step128","best_objective","best_position_error"):
        print(json.dumps(row),flush=True)

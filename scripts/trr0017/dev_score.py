from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
import native as n
from collections import defaultdict
name=sys.argv[1]
X=ROOT/"experiments/TRR-0017"
f=json.loads((X/(name+"_freeze.json")).read_text())
for e in f["entries"]:
    assert n.digest(ROOT/e["path"])==e["sha256"]
# All prediction files exist and are immutable before reading any labels.
truthpath=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1/evaluator_truth.json"
truth=json.loads(truthpath.read_text());groups=defaultdict(list);rr=[]
for e in f["entries"]:
    p=n.load_file(str(ROOT/e["path"]));t=n.torch.tensor(truth[e["id"]])
    for metric,v in p.items():
        ok=v[1:]==t[1:]
        r={k:e[k] for k in ("id","condition","group","method","total")}
        r.update(metric=metric,correct=int(ok.sum()),scored=len(ok),exact=bool(ok.all()))
        rr.append(r);groups[(r["condition"],r["group"],r["method"],metric)].append(r)
summary=[]
for k,rows in groups.items():
    s=dict(zip(("condition","group","method","projection"),k))
    s.update(correct=sum(r["correct"] for r in rows),scored=sum(r["scored"] for r in rows),
             exact=sum(r["exact"] for r in rows),records=len(rows),
             mean_seconds=sum(r["total"] for r in rows)/len(rows))
    summary.append(s)
n.write(X/(name+"_score.json"),{"summary":summary,"rows":rr,"truth_sha256":n.digest(truthpath),
         "freeze_sha256":n.digest(X/(name+"_freeze.json")),"scope":"opened retrospective development only"})
print(json.dumps(summary,indent=2))


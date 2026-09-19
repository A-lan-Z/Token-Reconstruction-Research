"""Verify and archive completed development evidence without opening truth."""
from pathlib import Path
import json,hashlib,zipfile,sys
ROOT=Path(__file__).resolve().parents[1]
X=ROOT/"experiments/TRR-0020"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def archive(stage):
    freeze=X/(stage+"_freeze.json")
    f=json.loads(freeze.read_text())
    for name,sha in f["binding"]["sources"].items():
        if digest(ROOT/name)!=sha:raise RuntimeError("changed bound source "+name)
    paths=set()
    for e in f["entries"]:
        p=ROOT/e["path"]
        if digest(p)!=e["sha256"]:raise RuntimeError("changed frozen output")
        paths.update([p,p.with_suffix(".json")])
    paths.update(X.glob(stage+"_qualification_*.safetensors"))
    target=X/(stage+"_predictions.zip");members=[]
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            name=p.relative_to(ROOT).as_posix();z.write(p,name)
            members.append({"path":name,"sha256":digest(p),"bytes":p.stat().st_size})
    with zipfile.ZipFile(target) as z:
        for e in members:
            if hashlib.sha256(z.read(e["path"])).hexdigest()!=e["sha256"]:raise RuntimeError("archive verification failed")
    receipt={"archive":str(target.relative_to(ROOT)),"sha256":digest(target),"bytes":target.stat().st_size,
             "complete_matrix":True,"cells":len(f["entries"]),"freeze_sha256":digest(freeze),"members":members}
    with (X/(stage+"_archive.json")).open("x") as out:json.dump(receipt,out,indent=2)
    print(stage,len(f["entries"]),len(members),target.stat().st_size,flush=True)
if __name__=="__main__":
    for stage in sys.argv[1:]:archive(stage)

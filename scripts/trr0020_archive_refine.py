"""Archive the complete canonical experiment after its freeze, scoring and guard pass."""
from pathlib import Path
import sys,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/trr0020_canonical_refine"))
from support import n,X,OUT,METHODS
import importlib.util
_spec=importlib.util.spec_from_file_location("trr0020_refine_score",ROOT/"scripts/trr0020_canonical_refine/score.py")
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
gate=_module.gate

def main():
    f=json.loads((X/"prediction_freeze.json").read_text())
    rows=json.loads((X/"metadata.json").read_text())
    outputs,runs=gate(f,rows)
    matrix=json.loads((X/"canonical_matrix.json").read_text())
    guard=json.loads((X/"guard.json").read_text())
    if not matrix["complete"] or matrix["required_cells"]!=68:raise RuntimeError("incomplete canonical matrix")
    if guard["returncode"]!=0 or guard["failure"] is not None:raise RuntimeError("failed guard")
    paths={ROOT/e["path"] for e in f["entries"]}
    for e in f["entries"]:
        receipt=OUT/"receipts"/f'{e["id"]}__{e["method"]}__r{e["rep"]}.json'
        if json.loads(receipt.read_text())!=e:raise RuntimeError("receipt differs from freeze")
        paths.add(receipt)
    for receipt in (X/"qualifications").glob("*.json"):
        q=json.loads(receipt.read_text())
        if q["binding"]!=f["binding"]:raise RuntimeError("qualification binding changed")
        for r in q["runs"]:
            p=ROOT/r["path"]
            if n.digest(p)!=r["sha256"]:raise RuntimeError("qualification output changed")
            paths.add(p)
        paths.add(receipt)
    required=["prediction_freeze.json","score.json","canonical_matrix.json","truth_gate.json","guard.json",
      "BENCHMARK_PLAN.md","registry.json","metadata.json","inherited_matrix.json","inherited_registry.json",
      "validation.json","validation_guard.json","validation_outputs.zip","validation_archive.json"]
    paths.update(X/name for name in required)
    paths.update(ROOT/phase["path"] for phase in f["phases"])
    target=X/"raw_evidence.zip";members=[]
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            name=p.relative_to(ROOT).as_posix()
            z.write(p,name)
            members.append({"path":name,"sha256":n.digest(p),"bytes":p.stat().st_size})
    with zipfile.ZipFile(target) as z:
        for e in members:
            if hashlib.sha256(z.read(e["path"])).hexdigest()!=e["sha256"]:
                raise RuntimeError("archive round-trip failed")
    n.write(X/"archive.json",{"archive":str(target.relative_to(ROOT)),"sha256":n.digest(target),
      "bytes":target.stat().st_size,"members":members,"replicate_cells":len(f["entries"]),
      "unique_outputs":len(outputs),"complete_matrix_cells":68,"round_trip_verified":True,
      "freeze_sha256":n.digest(X/"prediction_freeze.json"),"score_sha256":n.digest(X/"score.json")})
    # Split the portable archive into sub-50MiB git-sized parts; retain full SHA for reassembly.
    parts=[]
    with target.open("rb") as inp:
        index=1
        while True:
            data=inp.read(48*2**20)
            if not data:break
            part=X/("raw_evidence.zip.part"+str(index).zfill(2))
            with part.open("xb") as out:out.write(data)
            parts.append({"path":str(part.relative_to(ROOT)),"sha256":n.digest(part),"bytes":len(data)})
            index+=1
    n.write(X/"archive_parts.json",{"assembled_sha256":n.digest(target),"assembled_bytes":target.stat().st_size,
      "parts":parts,"restore":"Concatenate parts in the listed order; verify assembled SHA256 before extracting."})
    print("ARCHIVED",len(f["entries"]),len(outputs),len(members),target.stat().st_size,flush=True)
if __name__=="__main__":main()

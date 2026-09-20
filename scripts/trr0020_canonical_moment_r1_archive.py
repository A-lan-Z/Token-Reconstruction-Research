"""Archive exact raw evidence for the moment canonical comparison and original qualification."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
X=ROOT/"experiments/TRR-0020/canonical_moment_r1"
OLD=ROOT/"experiments/TRR-0020/canonical_moment"
OUT=ROOT/"outputs/TRR-0020/canonical_moment_r1"
def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    started=time.time();paths={}
    def add(path,expected=None):
        p=ROOT/path
        if not p.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError("unexpected external member")
        sha=digest(p)
        if expected is not None and sha!=expected:raise RuntimeError("changed evidence "+path)
        item={"path":path,"sha256":sha,"size":p.stat().st_size,"archive_member":"objects/"+sha}
        if path in paths and paths[path]!=item:raise RuntimeError("inconsistent duplicate")
        paths[path]=item
    freeze=json.loads((X/"prediction_freeze.json").read_text())
    score=json.loads((X/"score.json").read_text())
    guard=json.loads((X/"matrix_guard.json").read_text())
    if len(freeze["entries"])!=1728 or guard["returncode"]!=0 or guard["failure"] is not None:raise RuntimeError("incomplete matrix")
    if score["freeze_sha256"]!=digest(X/"prediction_freeze.json"):raise RuntimeError("score binding")
    for e in freeze["entries"]:
        add(e["path"],e["sha256"])
        rp=OUT/"receipts"/(e["id"]+"__"+e["method"]+"__r"+str(e["rep"])+".json")
        if json.loads(rp.read_text())!=e:raise RuntimeError("receipt differs from freeze")
        add(str(rp.relative_to(ROOT)))
    for phase in freeze["phases"]:add(phase["path"],phase["sha256"])
    qualification_runs=0
    for method in ["a1_native","moment64","a1_graph"]:
        for rep in range(3):
            receipts=list((X/"qualifications").glob(f"receipt_{method}_{rep}_*.json"))
            if len(receipts)!=1:raise RuntimeError("qualification receipt count")
            p=receipts[0];q=json.loads(p.read_text());add(str(p.relative_to(ROOT)))
            if len(q["runs"])!=3:raise RuntimeError("qualification repetition count")
            for e in q["runs"]:add(e["path"],e["sha256"]);qualification_runs+=1
    validation_checks=0
    for folder in [OLD,X]:
        p=folder/"validation.json";v=json.loads(p.read_text());add(str(p.relative_to(ROOT)))
        if not v["passed"] or len(v["checks"])!=52:raise RuntimeError("validation incomplete")
        for e in v["checks"]:
            add(e["path"],e["sha256"])
            rp=ROOT/e["path"];rp=rp.with_suffix(".json")
            if json.loads(rp.read_text())!=e:raise RuntimeError("validation receipt mismatch")
            add(str(rp.relative_to(ROOT)));validation_checks+=1
    for name in ["matrix_failure.json","matrix_terminal_stderr.txt","matrix_launch.json","matrix_guard.json","validation_launch.json","validation_guard.json"]:
        add(str((OLD/name).relative_to(ROOT)))
    logical=[paths[k] for k in sorted(paths)];objects={}
    for e in logical:objects.setdefault(e["archive_member"],e)
    archive=OUT/"raw_evidence.zip"
    if archive.exists() or (X/"archive.json").exists():raise RuntimeError("create-only archive exists")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for member,e in sorted(objects.items()):z.write(ROOT/e["path"],member)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(objects):raise RuntimeError("archive membership")
        for member,e in objects.items():
            value=z.read(member)
            if len(value)!=e["size"] or hashlib.sha256(value).hexdigest()!=e["sha256"]:raise RuntimeError("archive roundtrip")
    parts=[];combined=hashlib.sha256()
    with archive.open("rb") as source:
        for index in range(1,1000):
            value=source.read(48*2**20)
            if not value:break
            p=X/f"evidence.zip.part{index:03d}"
            with p.open("xb") as target:target.write(value)
            combined.update(p.read_bytes())
            parts.append({"path":str(p.relative_to(ROOT)),"size":len(value),"sha256":digest(p)})
    full=digest(archive)
    if combined.hexdigest()!=full:raise RuntimeError("part reassembly")
    result={"task_id":"TRR-0020","format":"sha256-object-archive-v1","logical_members":logical,
      "logical_member_count":len(logical),"unique_objects":len(objects),"raw_archive_bytes":archive.stat().st_size,
      "archive_sha256":full,"parts":parts,"all_members_roundtrip_verified":True,
      "current_matrix_replicate_cells":1728,"current_largest_case_runs":qualification_runs,
      "original_and_R1_validation_checks":validation_checks,
      "freeze_sha256":digest(X/"prediction_freeze.json"),"score_sha256":digest(X/"score.json"),
      "reassembly":"Concatenate parts in listed order; verify archive_sha256; logical_members maps paths to ZIP objects.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":started,"end_unix":time.time()}
    with (X/"archive.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="logical_members"},indent=2))
if __name__=="__main__":main()

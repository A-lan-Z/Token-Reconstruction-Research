"""Archive and verify every raw development43 fused scalar decoder artifact with exact-byte deduplication."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
X=ROOT/"experiments/TRR-0020"
def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    started=time.time();paths={}
    def add(path,expected=None):
        p=ROOT/path
        sha=digest(p)
        if expected is not None and sha!=expected:raise RuntimeError("changed raw evidence "+path)
        paths[path]={"path":path,"sha256":sha,"size":p.stat().st_size,"archive_member":"objects/"+sha}
    f=json.loads((X/"dev43_freeze.json").read_text())
    for entry in f["entries"]:
        for rep in entry["repetitions"]:add(rep["path"],rep["sha256"])
        receipt=Path(entry["path"]).parent/(entry["id"]+"__"+entry["method"]+".json")
        add(str(receipt))
    for phase in f["phases"]:
        p=json.loads((ROOT/phase["path"]).read_text())
        if digest(ROOT/phase["path"])!=phase["sha256"]:raise RuntimeError("changed phase")
        q=json.loads((ROOT/p["qualification_path"]).read_text())
        if digest(ROOT/p["qualification_path"])!=p["qualification_sha256"]:raise RuntimeError("changed qualification")
        for entry in q["runs"]+q["eager"]:add(entry["path"],entry["sha256"])
    logical=[paths[key] for key in sorted(paths)]
    objects={}
    for entry in logical:objects.setdefault(entry["archive_member"],entry)
    archive=ROOT/"outputs/TRR-0020/dev43_raw_evidence.zip"
    if archive.exists() or (X/"dev43_archive.json").exists():raise RuntimeError("create-only archive exists")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for member,entry in sorted(objects.items()):z.write(ROOT/entry["path"],member)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(objects):raise RuntimeError("archive membership")
        for member,entry in objects.items():
            payload=z.read(member)
            if len(payload)!=entry["size"] or hashlib.sha256(payload).hexdigest()!=entry["sha256"]:raise RuntimeError("archive roundtrip")
    parts=[];combined=hashlib.sha256()
    with archive.open("rb") as source:
        for index in range(1,1000):
            payload=source.read(48*2**20)
            if not payload:break
            p=X/f"dev43_evidence.zip.part{index:03d}"
            with p.open("xb") as target:target.write(payload)
            combined.update(p.read_bytes())
            parts.append({"path":str(p.relative_to(ROOT)),"size":len(payload),"sha256":digest(p)})
    full_sha=digest(archive)
    if combined.hexdigest()!=full_sha:raise RuntimeError("part reassembly differs")
    record={"task_id":"TRR-0020","format":"sha256-object-archive-v1","logical_members":logical,
      "logical_member_count":len(logical),"unique_objects":len(objects),"raw_archive_bytes":archive.stat().st_size,
      "archive_sha256":full_sha,"parts":parts,"all_members_roundtrip_verified":True,
      "reassembly":"Concatenate parts in listed order; verify archive_sha256; logical_members maps original paths to ZIP members.",
      "freeze_sha256":digest(X/"dev43_freeze.json"),"score_sha256":digest(X/"dev43_score.json"),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":started,"end_unix":time.time()}
    with (X/"dev43_archive.json").open("x") as out:json.dump(record,out,indent=2);out.write("\n")
    print(json.dumps({k:v for k,v in record.items() if k!="logical_members"},indent=2))
if __name__=="__main__":main()

"""Create a hash-deduplicated portable archive of development39 public homotopy evidence."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    started=time.time();q=json.loads((X/"dev39_gpu_qualification.json").read_text())
    if not q["passed"] or len(q["derivative_cells"])!=20 or len(q["linear_cells"])!=60:raise RuntimeError("incomplete qualification")
    entries=q["fixtures"]+q["derivative_cells"]+[rep for cell in q["linear_cells"] for rep in cell["repetitions"]]
    entries=list({v["path"]:v for v in entries}.values())
    logical=[];objects={}
    for row in entries:
        path=ROOT/row["path"];sha=digest(path)
        if sha!=row["sha256"]:raise RuntimeError("raw hash changed")
        entry={"path":row["path"],"sha256":sha,"size":path.stat().st_size,"archive_member":"objects/"+sha}
        logical.append(entry);objects.setdefault(entry["archive_member"],entry)
    archive=ROOT/"outputs/TRR-0020/dev39_gpu_evidence.zip"
    if archive.exists() or (X/"dev39_gpu_archive.json").exists():raise RuntimeError("create-only archive exists")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for member,row in sorted(objects.items()):z.write(ROOT/row["path"],member)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(objects):raise RuntimeError("archive members differ")
        for member,row in objects.items():
            data=z.read(member)
            if len(data)!=row["size"] or hashlib.sha256(data).hexdigest()!=row["sha256"]:raise RuntimeError("archive roundtrip")
    parts=[];combined=hashlib.sha256()
    with archive.open("rb") as source:
        for index in range(1,1000):
            payload=source.read(48*2**20)
            if not payload:break
            part=X/f"dev39_gpu_evidence.zip.part{index:03d}"
            with part.open("xb") as f:f.write(payload)
            combined.update(part.read_bytes());parts.append({"path":str(part.relative_to(ROOT)),"size":len(payload),"sha256":digest(part)})
    sha=digest(archive)
    if combined.hexdigest()!=sha:raise RuntimeError("part reassembly")
    result={"task_id":"TRR-0020","format":"sha256-object-archive-v1","logical_members":logical,
      "logical_member_count":len(logical),"unique_objects":len(objects),"raw_archive_bytes":archive.stat().st_size,
      "archive_sha256":sha,"parts":parts,"all_members_roundtrip_verified":True,
      "reassembly":"Concatenate parts in order; verify archive_sha256. logical_members maps original paths to ZIP objects.",
      "qualification_sha256":digest(X/"dev39_gpu_qualification.json"),"archive_script_sha256":digest(Path(__file__)),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":started,"end_unix":time.time()}
    with (X/"dev39_gpu_archive.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="logical_members"}))
if __name__=="__main__":main()

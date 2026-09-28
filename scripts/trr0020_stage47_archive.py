"""Preserve failed and accepted public history-mixing raw evidence separately."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess,sys,argparse
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--attempt",choices=["failed","r1"],required=True);args=parser.parse_args()
    start=time.time();entries=[]
    if args.attempt=="failed":
        failure=json.loads((X/"dev47_failure.json").read_text())
        entries=[{"path":s} for s in failure["raw_paths"]]+[{"path":"experiments/TRR-0020/"+failure["partial_receipt"]}]
        prefix="dev47_failed"
    else:
        prefix="dev47_r1"
        freeze=json.loads((X/"dev47_r1_public_freeze.json").read_text())
        if len(freeze["phases"])!=4:raise RuntimeError("incomplete")
        for phase in freeze["phases"]:
            path=ROOT/phase["path"]
            if sha(path)!=phase["sha256"]:raise RuntimeError("phase changed")
            q=json.loads(path.read_text())
            if not q["passed"]:raise RuntimeError("unqualified")
            entries.extend(q["runs"]+q["eager"])
    logical=[];objects={}
    for row in entries:
        path=ROOT/row["path"];digest=sha(path)
        if row.get("sha256",digest)!=digest:raise RuntimeError("raw changed")
        entry={"path":row["path"],"sha256":digest,"size":path.stat().st_size,"archive_member":"objects/"+digest}
        logical.append(entry);objects.setdefault(digest,entry)
    archive=ROOT/"outputs/TRR-0020"/(prefix+"_evidence.zip")
    if archive.exists() or (X/(prefix+"_archive.json")).exists():raise RuntimeError("create-only")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for digest,row in sorted(objects.items()):z.write(ROOT/row["path"],row["archive_member"])
    with zipfile.ZipFile(archive) as z:
        for row in objects.values():
            data=z.read(row["archive_member"])
            if len(data)!=row["size"] or hashlib.sha256(data).hexdigest()!=row["sha256"]:raise RuntimeError("roundtrip")
    parts=[];combined=hashlib.sha256()
    with archive.open("rb") as z:
        for index in range(1,100):
            payload=z.read(48*2**20)
            if not payload:break
            path=X/(prefix+f"_evidence.zip.part{index:03d}")
            with path.open("xb") as f:f.write(payload)
            parts.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"size":len(payload)});combined.update(payload)
    if combined.hexdigest()!=sha(archive):raise RuntimeError("reassembly")
    result={"task_id":"TRR-0020","attempt":args.attempt,"format":"sha256-object-archive-v1","logical_members":logical,
      "logical_member_count":len(logical),"unique_objects":len(objects),"raw_archive_bytes":archive.stat().st_size,
      "archive_sha256":sha(archive),"parts":parts,"all_members_roundtrip_verified":True,
      "reassembly":"Concatenate listed parts; verify archive_sha256; logical_members maps paths to ZIP members.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":start,"end_unix":time.time()}
    with (X/(prefix+"_archive.json")).open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="logical_members"}))
if __name__=="__main__":main()

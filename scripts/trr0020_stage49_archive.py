"""Archive every qualified moment-bound update with exact-byte deduplication."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def digest(p):
    with Path(p).open("rb") as h:return hashlib.file_digest(h,"sha256").hexdigest()
def main():
    started=time.time();r=json.loads((X/"dev49_r1_gpu_check.json").read_text())
    if not r["passed"]:raise RuntimeError("unqualified")
    logical=[]
    for cell in r["cells"]:
      for rep in cell["repetitions"]:
        p=ROOT/rep["path"]
        if digest(p)!=rep["sha256"]:raise RuntimeError("changed output")
        logical.append({"path":rep["path"],"sha256":rep["sha256"],"size":p.stat().st_size,"archive_member":"objects/"+rep["sha256"]})
    objects={}
    for entry in logical:objects.setdefault(entry["archive_member"],entry)
    archive=ROOT/"outputs/TRR-0020/dev49_r1_raw_evidence.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,entry in sorted(objects.items()):z.write(ROOT/entry["path"],name)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(objects):raise RuntimeError("membership")
        for name,entry in objects.items():
            payload=z.read(name)
            if len(payload)!=entry["size"] or hashlib.sha256(payload).hexdigest()!=entry["sha256"]:raise RuntimeError("roundtrip")
    parts=[];combined=hashlib.sha256()
    with archive.open("rb") as source:
      for index in range(1,1000):
        payload=source.read(48*2**20)
        if not payload:break
        p=X/f"dev49_r1_evidence.zip.part{index:03d}"
        with p.open("xb") as h:h.write(payload)
        combined.update(p.read_bytes());parts.append({"path":str(p.relative_to(ROOT)),"size":len(payload),"sha256":digest(p)})
    if combined.hexdigest()!=digest(archive):raise RuntimeError("reassembly")
    result={"task_id":"TRR-0020","logical_members":logical,"logical_member_count":len(logical),"unique_objects":len(objects),
      "all_members_roundtrip_verified":True,"raw_archive_bytes":archive.stat().st_size,"archive_sha256":digest(archive),"parts":parts,
      "qualification_sha256":digest(X/"dev49_r1_gpu_check.json"),"reassembly":"Concatenate parts in listed order; verify archive_sha256; logical_members maps paths to ZIP objects.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":started,"end_unix":time.time()}
    with (X/"dev49_r1_archive.json").open("x") as h:json.dump(result,h,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ["logical_members","parts"]}),flush=True)
if __name__=="__main__":main()

"""Reuse a verified prior raw archive when all new evidence bytes already exist there."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    start=time.time();q=json.loads((X/"dev46_gpu_check.json").read_text());base=json.loads((X/"dev44_gpu_archive.json").read_text())
    if not q["passed"] or len(q["cells"])!=20 or len(q["benchmarks"])!=8:raise RuntimeError("incomplete")
    if not base["all_members_roundtrip_verified"]:raise RuntimeError("unverified base")
    old={v["sha256"]:v for v in base["logical_members"]}
    entries=[v["public_fixture"] for v in q["cells"] if v["public_fixture"] is not None]+[r for v in q["cells"] for r in v["repetitions"]]
    logical=[];seen=set()
    for v in entries:
        if v["path"] in seen:continue
        seen.add(v["path"]);p=ROOT/v["path"];sha=digest(p)
        if sha!=v["sha256"] or sha not in old or p.stat().st_size!=old[sha]["size"]:raise RuntimeError("not byte-identical to archived evidence")
        logical.append({"path":v["path"],"sha256":sha,"size":p.stat().st_size,"archive_member":old[sha]["archive_member"]})
    archive=ROOT/"outputs/TRR-0020/dev44_gpu_evidence.zip"
    if digest(archive)!=base["archive_sha256"]:raise RuntimeError("base ZIP hash changed")
    with zipfile.ZipFile(archive) as z:
        for v in {v["sha256"]:v for v in logical}.values():
            data=z.read(v["archive_member"])
            if len(data)!=v["size"] or hashlib.sha256(data).hexdigest()!=v["sha256"]:raise RuntimeError("base roundtrip")
    for part in base["parts"]:
        if digest(ROOT/part["path"])!=part["sha256"]:raise RuntimeError("base part changed")
    result={"task_id":"TRR-0020","format":"sha256-object-archive-v1","logical_members":logical,"logical_member_count":len(logical),
      "unique_objects":len({v["sha256"] for v in logical}),"raw_archive_bytes":base["raw_archive_bytes"],
      "new_archive_bytes":0,"archive_sha256":base["archive_sha256"],"parts":base["parts"],
      "base_archive":{"path":"experiments/TRR-0020/dev44_gpu_archive.json","sha256":digest(X/"dev44_gpu_archive.json")},
      "all_members_roundtrip_verified":True,"qualification_sha256":digest(X/"dev46_gpu_check.json"),
      "reassembly":"Reuse listed verified development44 parts; concatenate, verify archive_sha256, then map new logical paths to existing ZIP object members.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":start,"end_unix":time.time()}
    with (X/"dev46_gpu_archive.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["logical_members","parts"]}))
if __name__=="__main__":main()

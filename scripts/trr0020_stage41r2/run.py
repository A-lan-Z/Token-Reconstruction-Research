"""One isolated process per native profiling geometry; no overlap."""
from pathlib import Path
import json,subprocess,sys,time,hashlib,os
ROOT=Path(__file__).resolve().parents[2];X=ROOT/"experiments/TRR-0020";DEST=X/"dev41_profile_r2.json"
if DEST.exists():raise RuntimeError("create-only")
start=time.time();workers=[]
for length in [128,40]:
    command=[sys.executable,str(Path(__file__).parent/"profile_current.py"),"--length",str(length)]
    subprocess.run(command,cwd=ROOT,check=True)
    p=X/f"dev41_profile_r2_{length}.json";d=json.loads(p.read_text())
    if not d["passed"]:raise RuntimeError("failed worker")
    workers.append({"length":length,"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"result":d})
result={"task_id":"TRR-0020","scope":"public execution profile; isolated native geometries; no new benchmark method",
 "passed":True,"workers":[{k:v for k,v in w.items() if k!="result"} for w in workers],
 "runs":[v for w in workers for v in w["result"]["runs"]],"profiles":[v for w in workers for v in w["result"]["profiles"]],
 "update_fixtures":[v for w in workers for v in w["result"]["update_fixtures"]],
 "prefix_seconds":sum(w["result"]["prefix_seconds"] for w in workers),"engine_seconds":sum(w["result"]["engine_seconds"] for w in workers),
 "peak_reserved":max(w["result"]["peak_reserved"] for w in workers),"peak_allocated":max(w["result"]["peak_allocated"] for w in workers),
 "start_unix":start,"end_unix":time.time(),"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
 "command":[sys.executable,*sys.argv],"truth_read":False,"shortlist":None}
with DEST.open("x") as f:json.dump(result,f,indent=2)
print("BOTH_ISOLATED_PROFILES_COMPLETE",flush=True)

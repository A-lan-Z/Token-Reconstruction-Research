"""Regression check: baseline imports must not inherit novel-method arithmetic."""
from pathlib import Path
import subprocess,sys,json,os,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
X=ROOT/"experiments/TRR-0020/canonical_moment_r1"
CODE=r"""
import sys,os,json,hashlib
sys.path.insert(0,"scripts/trr0020_canonical_moment_r1")
import predict as worker
import moment_canonical_support as support
torch=support.torch
novel=sys.argv[1]=="novel"
if novel: support.get_development()
before=(os.environ.get("CUBLAS_WORKSPACE_CONFIG"),torch.are_deterministic_algorithms_enabled(),torch.cuda.is_initialized())
assert before==((":4096:8",True,False) if novel else (None,False,False)),before
bound=support.binding()
after=(os.environ.get("CUBLAS_WORKSPACE_CONFIG"),torch.are_deterministic_algorithms_enabled(),torch.cuda.is_initialized())
assert after==before,after
assert ("grid50_support" in sys.modules)==novel
print(json.dumps({"mode":sys.argv[1],"before":before,"after":after,"development_loaded_in_parent":novel,
 "binding_sha256":hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest(),"sources":bound["sources"]}))
"""
def main():
    started=time.time();checks=[]
    for mode in ["baseline","novel"]:
        env={**os.environ,"PYTHONPATH":"src","PYTHONDONTWRITEBYTECODE":"1"}
        if mode=="novel":env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        else:env.pop("CUBLAS_WORKSPACE_CONFIG",None)
        command=[sys.executable,"-c",CODE,mode]
        entry=json.loads(subprocess.check_output(command,cwd=ROOT,env=env,text=True))
        checks.append(entry)
    assert checks[0]["binding_sha256"]==checks[1]["binding_sha256"]
    result={"task_id":"TRR-0020","passed":True,"checks":checks,"cuda_initialized":False,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":started,"end_unix":time.time()}
    with (X/"import_check.json").open("x") as f:json.dump(result,f,indent=2)
    print("BOTH_ENVIRONMENTS_ISOLATED_NO_CUDA",checks[0]["binding_sha256"],flush=True)
if __name__=="__main__":main()

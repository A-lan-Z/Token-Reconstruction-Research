from support import *
import os
def main():
    X.mkdir(parents=True,exist_ok=True);bound=binding()
    for rep in range(3):
        order=METHODS[rep:]+METHODS[:rep]
        for method in order:
            env={**os.environ,"PYTHONPATH":"src","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"}
            if method in METHOD_IDS:env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
            else:env.pop("CUBLAS_WORKSPACE_CONFIG",None)
            command=[sys.executable,str(ROOT/"scripts/trr0020_canonical/predict.py"),"--method",method,"--rep",str(rep)]
            print("PHASE_START",method,rep,utc(),flush=True)
            subprocess.run(command,cwd=ROOT,env=env,check=True)
    phases=[];entries=[]
    for rep in range(3):
        for method in METHODS:
            p=X/f"phase_{method}_{rep}.json";f=json.loads(p.read_text());phase_gate(f,bound)
            phases.append({"path":str(p.relative_to(ROOT)),"sha256":n.digest(p)});entries.extend(f["entries"])
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(X/"prediction_freeze.json",{"task_id":"TRR-0020","binding":bound,"phases":phases,"entries":entries,
      "truth_read":False,"total_replicate_cells":len(entries),"completed_utc":utc()})
    print("FULL2304RUNFREEZE_COMPLETE",flush=True)
if __name__=="__main__":main()

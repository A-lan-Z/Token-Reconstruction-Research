"""Independent constrained two-column least-squares reference for history mixing."""
from pathlib import Path
import sys,json,time,subprocess,hashlib,torch
from aa_step import mix
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/"experiments/TRR-0020/dev47_cpu_reference.json"
def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.manual_seed(47047);g=torch.Generator().manual_seed(47047)
    result={"task_id":"TRR-0020","passed":False,"cases":[],"start_unix":time.time(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"environment":{"torch":torch.__version__,"python":sys.version},
      "sources":{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(Path(__file__).parent.glob("*.py"))+[ROOT/"experiments/TRR-0020/DEV47_PLAN.md"]}}
    try:
      for dtype in [torch.float32,torch.float64]:
       for metric in ["logit","probability"]:
        for kind in ["random","identical_residual","zero_residual","no_history"]:
          current=torch.randn(3,17,dtype=dtype,generator=g);proposal=torch.randn(3,17,dtype=dtype,generator=g)
          probability=current.softmax(-1);r=proposal-current if metric=="logit" else proposal.softmax(-1)-probability
          old=torch.randn_like(r);previous=torch.randn_like(proposal)
          if kind=="identical_residual":old=r.clone()
          if kind=="zero_residual":proposal=current.clone();r.zero_();old.zero_()
          valid=torch.tensor(kind!="no_history")
          out,residual,info=mix(current,proposal,probability,old,previous,valid,metric,1.)
          rd=residual.double().reshape(-1);od=old.double().reshape(-1);lam=1e-4*(rd.dot(rd)+od.dot(od))
          if kind in ["no_history","zero_residual"]:coef=0.
          else:
            columns=torch.stack([od,rd],1);gram=columns.T@columns;gram[0,0]+=lam
            kkt=torch.zeros(3,3,dtype=torch.float64);kkt[:2,:2]=gram;kkt[:2,2]=1;kkt[2,:2]=1
            rhs=torch.tensor([0.,0.,1.],dtype=torch.float64)
            coef=float(torch.linalg.solve(kkt,rhs)[0].clamp(-1,1))
          expected=proposal.double()-coef*(proposal.double()-previous.double())
          expected=expected-expected.amax(-1,keepdim=True) if kind not in ["no_history","zero_residual"] else proposal.double()
          error=float((out.double()-expected).abs().max());ce=abs(float(info["coefficient"])-coef)
          tolerance=2e-3 if dtype==torch.float32 else 1e-9
          passed=error<tolerance and ce<tolerance
          if kind=="no_history":passed=passed and torch.equal(out,proposal)
          result["cases"].append({"dtype":str(dtype),"metric":metric,"kind":kind,"coefficient_error":ce,"output_error":error,"passed":passed})
          if not passed:raise RuntimeError("independent AA reference failed")
      result["passed"]=len(result["cases"])==16
    except Exception:
      import traceback
      result["failure"]=traceback.format_exc();raise
    finally:
      result["end_unix"]=time.time()
      with DEST.open("x") as f:json.dump(result,f,indent=2)
      print(json.dumps({k:v for k,v in result.items() if k not in ["sources","cases"]}))
if __name__=="__main__":main()

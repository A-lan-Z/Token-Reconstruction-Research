from pathlib import Path
import torch,json,sys,time,subprocess,hashlib,traceback
from aa_step import mix
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/"experiments/TRR-0020/dev48_r1_cpu_reference.json"
def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.manual_seed(48048);g=torch.Generator().manual_seed(48048)
    result={"task_id":"TRR-0020","passed":False,"cases":[],"start_unix":time.time(),"command":[sys.executable,*sys.argv],
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":{"torch":torch.__version__,"python":sys.version},
      "sources":{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(Path(__file__).parent.glob("*.py"))+[ROOT/"experiments/TRR-0020/DEV48_R1_PLAN.md"]}}
    try:
      for dtype in [torch.float32,torch.float64]:
       for metric in ["logit","probability"]:
        for clip in [1.,2.]:
         for kind in ["random","identical","zero","no_history"]+(["small","large"] if metric=="logit" else []):
          magnitude=1e-20 if kind=="small" else 1e20 if kind=="large" else 1.
          current=torch.randn(3,17,dtype=dtype,generator=g)*magnitude
          proposal=torch.randn(3,17,dtype=dtype,generator=g)*magnitude
          probability=current.softmax(-1)
          r=proposal-current if metric=="logit" else proposal.softmax(-1)-probability
          old=torch.randn(3,17,dtype=dtype,generator=g)*magnitude;previous=torch.randn(3,17,dtype=dtype,generator=g)*magnitude
          if kind=="identical":old=r.clone()
          if kind=="zero":proposal=current.clone();old.zero_()
          valid=torch.tensor(kind!="no_history")
          out,residual,info=mix(current,proposal,probability,old,previous,valid,metric,clip)
          rd=residual.double().reshape(-1);od=old.double().reshape(-1)
          norm=(rd.square().sum()+od.square().sum()).sqrt()
          if kind in ["zero","no_history"]:coef=0.
          else:
            rd=rd/norm;od=od/norm
            columns=torch.stack([od,rd],1);gram=columns.T@columns;gram[0,0]+=1e-4*(rd.dot(rd)+od.dot(od))
            kkt=torch.zeros(3,3,dtype=torch.float64);kkt[:2,:2]=gram;kkt[:2,2]=1;kkt[2,:2]=1
            coef=float(torch.linalg.solve(kkt,torch.tensor([0.,0.,1.],dtype=torch.float64))[0].clamp(-clip,clip))
          expected=proposal.double()-coef*(proposal.double()-previous.double())
          if kind not in ["zero","no_history"]:expected=expected-expected.amax(-1,keepdim=True)
          else:expected=proposal.double()
          err=float((out.double()-expected).abs().max()/max(magnitude,1e-30));ce=abs(float(info["coefficient"])-coef)
          tolerance=2e-3 if dtype==torch.float32 else 1e-9
          finite=bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(v).all()) for v in info.values())
          passed=finite and err<tolerance and ce<tolerance
          result["cases"].append({"dtype":str(dtype),"metric":metric,"clip":clip,"kind":kind,"scaled_output_error":err,"coefficient_error":ce,"finite":finite,"passed":passed})
          if not passed:raise RuntimeError("stable AA reference mismatch")
      current=torch.tensor([[-2e38,0.]],dtype=torch.float32);proposal=torch.tensor([[-3e38,0.]],dtype=torch.float32)
      old=torch.tensor([[-2e38,0.]],dtype=torch.float32);previous=torch.zeros_like(proposal)
      out,res,info=mix(current,proposal,current.softmax(-1),old,previous,torch.tensor(True),"logit",1.)
      fallback=torch.equal(out,proposal) and not bool(info["mixed_finite"]) and float(info["coefficient"])==0
      result["cases"].append({"kind":"explicit_mixed_overflow_fallback","passed":fallback})
      if not fallback:raise RuntimeError("overflow fallback")
      result["passed"]=len(result["cases"])==41
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      result["end_unix"]=time.time()
      with DEST.open("x") as f:json.dump(result,f,indent=2)
      print(json.dumps({k:v for k,v in result.items() if k not in ["sources","cases"]}))
if __name__=="__main__":main()

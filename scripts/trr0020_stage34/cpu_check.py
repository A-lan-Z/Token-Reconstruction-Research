"""Independent small nonlinear autograd and descent-invariant qualification."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess
import torch
from descent import solve,gradient,error,STEP_COLUMNS,TRIAL_COLUMNS
ROOT=Path(__file__).resolve().parents[2];X=ROOT/"experiments/TRR-0020"
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    start=time.time();torch.set_num_threads(2);cases=[];max_error=0.
    for dtype in [torch.float32,torch.float64]:
      for scale in [.2,2.,6.]:
       for seed in [34001,34002]:
        generator=torch.Generator().manual_seed(seed)
        E=torch.randn(23,6,generator=generator,dtype=dtype)
        W=torch.randn(6,9,generator=generator,dtype=dtype)*scale
        R=torch.randn(9,6,generator=generator,dtype=dtype)*.3
        z0=torch.randn(1,23,generator=generator,dtype=dtype)*.5
        def raw(x):return x+torch.tanh(x@W)@R
        target=raw(E[7:8])
        def evaluate(x):
            a=x@W;return x+torch.tanh(a)@R,a
        def vjp(v,cache):return v+(v@R.T*(1-torch.tanh(cache).square()))@W.T
        output,states,counts=solve(z0,target,E,evaluate,vjp,16)
        references=[]
        for z in [z0,output["logits"]]:
            p=z.softmax(-1).detach().requires_grad_();fresh=raw(p@E)
            loss=1-(torch.nn.functional.normalize(fresh,dim=-1)*torch.nn.functional.normalize(target,dim=-1)).sum()
            expected=torch.autograd.grad(loss,p)[0]
            mapped,cache=evaluate(p.detach()@E);actual,info=gradient(mapped,target,E,lambda v:vjp(v,cache))
            difference=float((actual-expected).abs().max());max_error=max(max_error,difference)
            torch.testing.assert_close(actual,expected,rtol=2e-5 if dtype==torch.float32 else 1e-10,atol=2e-5 if dtype==torch.float32 else 1e-10)
            references.append(difference)
        fresh=raw(output["logits"].softmax(-1)@E)
        torch.testing.assert_close(fresh,output["prefix_output"],rtol=0,atol=0)
        if not bool(torch.isfinite(output["logits"]).all()):raise RuntimeError("lost vocabulary coordinate")
        trace=output["step_trace"];trials=output["trial_trace"]
        tolerance=1e-7 if dtype==torch.float32 else 1e-12
        if not bool((trace[:,1]<=trace[:,0]+tolerance).all()):raise RuntimeError("accepted loss increased")
        previous=trace[:-1,1]
        torch.testing.assert_close(previous,trace[1:,0],rtol=0,atol=0)
        if counts["prefix_forwards"]!=1+len(states) or counts["prefix_vjps"]!=16:raise RuntimeError("work accounting")
        for old,new,budget in states:
            a=old.double().log_softmax(-1);b=new.double().log_softmax(-1)
            kl=(a.exp()*(a-b)).sum()
            if kl>float(budget[0])+2e-5 or kl < -2e-5:raise RuntimeError("KL budget exceeded")
        accepted=trials[:,7].bool()
        if not bool((trials[accepted,3]<=trials[accepted,8]).all()):raise RuntimeError("Armijo predicate")
        cases.append({"dtype":str(dtype),"scale":scale,"seed":seed,"gradient_errors":references,
          "initial_loss":float(trace[0,0]),"final_loss":float(trace[-1,1]),"counts":counts,
          "step_trace":trace.tolist(),"trial_trace":trials.tolist(),"passed":True})
    if not any(c["counts"]["rejected_trials"]>0 for c in cases):raise RuntimeError("rejection branch not exercised")
    if not all(c["counts"]["accepted_steps"]>0 for c in cases):raise RuntimeError("no successful progress")
    sources=[Path(__file__),Path(__file__).with_name("descent.py"),ROOT/"scripts/trr0020_stage32/budget_step.py",ROOT/"scripts/trr0020_stage33/probability_gradient.py"]
    result={"task_id":"TRR-0020","passed":True,"cases":cases,"case_count":len(cases),"gradient_max_absolute_error":max_error,
      "sources":{str(p.relative_to(ROOT)):sha(p) for p in sources},"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"environment":{"python":sys.version,"torch":torch.__version__},
      "start_unix":start,"end_unix":time.time(),"step_columns":STEP_COLUMNS,"trial_columns":TRIAL_COLUMNS}
    with (X/"dev34_descent_cpu.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({"passed":True,"cases":len(cases),"gradient_max_absolute_error":max_error,
      "rejected_trials":sum(c["counts"]["rejected_trials"] for c in cases),
      "accepted_steps":sum(c["counts"]["accepted_steps"] for c in cases)},indent=2))
if __name__=="__main__":main()

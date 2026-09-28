"""Independent algebra checks for tiled all-vocabulary scalar evaluation."""
from pathlib import Path
import sys,json,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"scripts/trr0020_stage32"))
import torch
from budget_step import update as original
from fused_step import update
DEST=ROOT/"experiments/TRR-0020/dev42_cpu_reference.json"

class TiledCPU:
    def __init__(self,lp,c,b,correction):
        self.lp=lp;self.c=c;self.base=b;self.correction=correction
    def __call__(self,t):
        chunks=[]
        for start in range(0,self.lp.shape[1],32):
            value=self.lp[:,start:start+32]-t[:,None]*self.c[:,start:start+32]
            m=value.max(-1).values;e=(value-m[:,None]).exp()
            chunks.append((m,e.sum(-1),(e*self.c[:,start:start+32]).sum(-1)))
        m=torch.stack([v[0] for v in chunks],-1);z=torch.stack([v[1] for v in chunks],-1);w=torch.stack([v[2] for v in chunks],-1)
        maximum=m.max(-1).values;scale=(m-maximum[:,None]).exp()
        total=(z*scale).sum(-1);weighted=(w*scale).sum(-1)
        return maximum+total.log()-self.base+t*self.correction,self.correction-weighted/total

def fixture(vocab,kind,dtype=torch.float64):
    g=torch.Generator().manual_seed(42042+vocab)
    logits=torch.randn(3,vocab,generator=g,dtype=dtype)*(50 if kind=="peaked" else 2)
    gradient=torch.randn(3,vocab,generator=g,dtype=dtype)
    if kind=="flat":gradient.fill_(.5)
    error=torch.tensor([0.,.02,.4],dtype=dtype)
    if kind=="zero_budget":error.zero_()
    return logits,gradient,error

def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    paths=list(Path(__file__).parent.glob("*.py"))+[ROOT/"scripts/trr0020_stage32/budget_step.py",ROOT/"experiments/TRR-0020/DEV42_PLAN.md"]
    result={"task_id":"TRR-0020","scope":"CPUFP64tiled algebra only","passed":False,"cases":[],
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv],
      "sources":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
      "environment":{"torch":torch.__version__,"python":sys.version},"start_unix":time.time()}
    try:
      for vocab in [17,1023,1031]:
       for kind in ["random","peaked","flat","zero_budget"]:
        logits,gradient,error=fixture(vocab,kind)
        lp=logits.log_softmax(-1);p=lp.exp();c=gradient-(p*gradient).sum(-1,keepdim=True)
        b=lp.logsumexp(-1);correction=(p*c).sum(-1)
        evaluator=TiledCPU(lp,c,b,correction);maximum=0.
        for rate in [0.,.1,5.,50.]:
            t=torch.full((3,),rate,dtype=torch.float64,requires_grad=True)
            reference=(lp-t[:,None]*c).logsumexp(-1)-b+t*correction
            derivative=torch.autograd.grad(reference.sum(),t)[0]
            value,d=evaluator(t)
            torch.testing.assert_close(value,reference,rtol=1e-10,atol=1e-10)
            torch.testing.assert_close(d,derivative,rtol=1e-10,atol=1e-10)
            maximum=max(maximum,float((value-reference).detach().abs().max()),float((d-derivative).detach().abs().max()))
        got,info=update(logits,gradient,error,2.,4,evaluator_factory=TiledCPU)
        expected,old=original(logits,gradient,error,2.,4)
        delta=(got.softmax(-1)-expected.softmax(-1)).abs().sum(-1).max()*.5
        if float(delta)>1e-9:raise RuntimeError("CPU update probability discrepancy")
        torch.testing.assert_close(got,expected,rtol=1e-8,atol=1e-8)
        if not torch.equal(info["active"],old["active"]):raise RuntimeError("active changed")
        result["cases"].append({"vocab":vocab,"kind":kind,"maximum_scalar_error":maximum,"maximum_logit_error":float((got-expected).abs().max()),"probability_TV":float(delta)})
      result["passed"]=len(result["cases"])==12
    except Exception:
      import traceback
      result["failure"]=traceback.format_exc();raise
    finally:
      result["end_unix"]=time.time()
      with DEST.open("x") as f:json.dump(result,f,indent=2)
      print(json.dumps({k:v for k,v in result.items() if k not in ["sources","cases"]}))
if __name__=="__main__":main()

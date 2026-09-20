"""Independent high-precision scalar roots and direct full-distribution KL."""
from pathlib import Path
import sys,json,time,subprocess,hashlib,traceback,math
import torch,mpmath as mp
from moment_step import update,MODES,RATE_MARGIN
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/"experiments/TRR-0020/dev49_cpu_reference.json"
KINDS=("random","peaked","flat","zero_budget","two_point","rare_winner","tiny_variance")
def fixture(vocab,kind,dtype=torch.float32):
    gen=torch.Generator().manual_seed(49049+vocab)
    z=torch.randn(8,vocab,generator=gen,dtype=dtype)*2
    g=torch.randn(8,vocab,generator=gen,dtype=dtype)
    if kind=="peaked":z*=25
    elif kind=="flat":g.fill_(.5)
    elif kind=="two_point":g=(torch.arange(vocab)%2).to(dtype).repeat(8,1)
    elif kind=="rare_winner":z.fill_(0.);z[:,0]=-18;g.fill_(1.);g[:,0]=0
    elif kind=="tiny_variance":z.fill_(-30);z[:,0]=0
    error=torch.tensor([-1e-7,0.,1e-10,1e-6,.005,.05,.5,1.],dtype=dtype)
    if kind=="zero_budget":error.zero_()
    return z,g,error
def reference_root(v,b,target,mode):
    v=mp.mpf(float(v));b=mp.mpf(float(b));target=mp.mpf(float(target));c=v/b**2
    def fn(s):
        if mode=="bennett":return mp.log(1+c*(mp.exp(s)-1-s))
        return mp.log((c*mp.exp(s)+mp.exp(-c*s))/(1+c))
    lower=mp.mpf(0);upper=mp.mpf(1)
    while fn(upper)<target:upper*=2
    for _ in range(110):
        middle=(lower+upper)/2
        if fn(middle)<=target:lower=middle
        else:upper=middle
    return float(lower/b*(1-RATE_MARGIN))
def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);mp.mp.dps=80
    sources=list(Path(__file__).parent.glob("*.py"))+[ROOT/"experiments/TRR-0020/DEV49_PLAN.md"]
    result={"task_id":"TRR-0020","passed":False,"start_unix":time.time(),"command":[sys.executable,*sys.argv],
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "sources":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
      "environment":{"torch":torch.__version__,"python":sys.version,"mpmath":mp.__version__},"cases":[]}
    try:
      for dtype in [torch.float32,torch.float64]:
       for vocab in [17,37,1031]:
        for kind in KINDS:
         z,g,e=fixture(vocab,kind,dtype)
         before=[v.clone() for v in [z,g,e]]
         for mode in MODES:
            out,info=update(z,g,e,mode,gpu_root=False)
            lp=z.double().log_softmax(-1);lq=out.double().log_softmax(-1)
            actual=(lp.exp()*(lp-lq)).sum(-1)
            tol=2e-10 if dtype==torch.float64 else 2e-5
            finite=all(bool(torch.isfinite(v).all()) for v in [out,*info.values()])
            feasible=bool((actual>=-tol).all() and (actual<=info["requested_budget"].double()+tol).all())
            relative=[]
            for row in range(len(z)):
                if info["active"][row] and info["target"][row]>=1e-8:
                    ref=reference_root(info["weighted_variance"][row],info["positive_bound"][row],info["target"][row],mode)
                    relative.append(abs(float(info["normalized_step"][row])-ref)/max(abs(ref),1e-300))
            preserved=all(torch.equal(a,b) for a,b in zip(before,[z,g,e]))
            row={"dtype":str(dtype),"vocab":vocab,"kind":kind,"mode":mode,"finite":finite,"feasible":feasible,
              "inputs_unchanged":preserved,"maximum_relative_root_error":max(relative,default=0),
              "direct_kl":actual.tolist(),"budget":info["requested_budget"].tolist(),"bound_kl":info["bound_kl"].tolist()}
            result["cases"].append(row)
            if not finite or not feasible or not preserved or max(relative,default=0)>3e-6:raise RuntimeError("CPU qualification failed")
      result["passed"]=len(result["cases"])==84
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      result["end_unix"]=time.time()
      with DEST.open("x") as f:json.dump(result,f,indent=2)
      print(json.dumps({k:v for k,v in result.items() if k not in ["sources","cases"]}))
if __name__=="__main__":main()

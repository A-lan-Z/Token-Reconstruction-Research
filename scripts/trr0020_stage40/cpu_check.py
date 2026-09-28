"""Independent dense public systems for the reduced-solver audit."""
from pathlib import Path
import sys,json,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage21"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage39"))
import torch
from homotopy import global_gmres
from krylov_audit import arnoldi,direction
DEST=ROOT/"experiments/TRR-0020/dev40_cpu_reference.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    paths=list(Path(__file__).parent.glob("*.py"))+[ROOT/"scripts/trr0020_stage39/homotopy.py",ROOT/"experiments/TRR-0020/DEV40_PLAN.md"]
    result={"task_id":"TRR-0020","execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"environment":{"torch":torch.__version__,"python":sys.version},
      "sources":{str(p.relative_to(ROOT)):sha(p) for p in paths},"start_unix":time.time(),"passed":False,"cases":[]}
    try:
      for dtype in [torch.float32,torch.float64]:
       for kind in ["well","nonnormal","ill","zero"]:
        g=torch.Generator().manual_seed(40040);d=24
        q=torch.linalg.qr(torch.randn(d,d,generator=g,dtype=dtype)).Q
        if kind=="ill":matrix=q@torch.diag(torch.logspace(-4,2,d,dtype=dtype))@q.T
        elif kind=="nonnormal":matrix=torch.eye(d,dtype=dtype)+3*torch.triu(torch.randn(d,d,generator=g,dtype=dtype),1)
        else:matrix=torch.eye(d,dtype=dtype)+.02*torch.randn(d,d,generator=g,dtype=dtype)
        b=torch.randn(3,8,generator=g,dtype=dtype)
        if kind=="zero":b.zero_()
        def mv(v):return (matrix@v.flatten()).reshape_as(v)
        basis,h,beta=arnoldi(mv,b,16)
        expected,_=global_gmres(mv,b,16,1e-6)
        original,_,_=direction(basis,h,beta,16,"normal_ridge")
        if not torch.equal(expected,original):raise RuntimeError("unchanged original differs")
        smallbasis,smallh,smallbeta=arnoldi(mv,b,8)
        if not torch.equal(basis[...,:9],smallbasis) or not torch.equal(h[:9,:8],smallh):raise RuntimeError("Arnoldi prefix changed")
        errors={}
        for mode in ["svd_ridge","svd_unregularized"]:
            answer,c,residual=direction(basis,h,beta,16,mode)
            hh=h.double();bb=torch.zeros(17,dtype=torch.float64);bb[0]=beta.double()
            if mode=="svd_ridge":
                scale=(h.T@h).diagonal().mean().clamp_min(1.).double()
                hh=torch.cat([hh,(1e-6*scale).sqrt()*torch.eye(16,dtype=torch.float64)])
                bb=torch.cat([bb,torch.zeros(16,dtype=torch.float64)])
            u,s,vh=torch.linalg.svd(hh,full_matrices=False)
            inv=torch.where(s>s.max()*1e-12,1/s.clamp_min(torch.finfo(s.dtype).tiny),0.)
            reference=vh.T@(inv*(u.T@bb))
            torch.testing.assert_close(c.double(),reference,rtol=3e-6 if dtype==torch.float32 else 2e-8,atol=3e-6 if dtype==torch.float32 else 2e-8)
            reference_direction=(basis[...,:16].double()*reference).sum(-1)
            torch.testing.assert_close(answer.double(),reference_direction,rtol=1e-4 if dtype==torch.float32 else 2e-7,atol=3e-5 if dtype==torch.float32 else 2e-7)
            if kind=="zero" and not torch.equal(answer,torch.zeros_like(answer)):raise RuntimeError("zero RHS")
            errors[mode]={"coefficient_maximum_error":float((c.double()-reference).abs().max()),
              "actual_relative_residual":float((mv(answer)-b).norm()/b.norm().clamp_min(1e-30))}
        result["cases"].append({"dtype":str(dtype),"kind":kind,"original_exact":True,"prefix_exact":True,"comparisons":errors})
      result["passed"]=len(result["cases"])==8
    except Exception:
      import traceback
      result["failure"]=traceback.format_exc();raise
    finally:
      result["end_unix"]=time.time()
      with DEST.open("x") as f:json.dump(result,f,indent=2)
      print(json.dumps({k:v for k,v in result.items() if k not in ["sources","cases"]}))
if __name__=="__main__":main()

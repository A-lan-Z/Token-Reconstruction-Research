"""Independent autograd checks of norm-preserving probability interpolation."""
from pathlib import Path
import torch,json,time,hashlib,subprocess,sys
from normalized_mixture import mixture,gradient,EPS
ROOT=Path(__file__).resolve().parents[2];X=ROOT/"experiments/TRR-0020"
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    start=time.time();torch.set_num_threads(2);cases=[];vertices=[]
    for dtype in [torch.float32,torch.float64]:
      for kind in ["regular","zero_mean","tiny_mean","zero_table"]:
       for nonlinear in [False,True]:
        generator=torch.Generator().manual_seed(35001)
        E=torch.randn(23,7,dtype=dtype,generator=generator);E[1]=-E[0]
        p=torch.randn(1,23,dtype=dtype,generator=generator).softmax(-1)
        if kind in ["zero_mean","tiny_mean"]:
            p.zero_();p[0,0]=.5;p[0,1]=.5
            if kind=="tiny_mean":E[1]=E[1]+1e-14
        if kind=="zero_table":E.zero_()
        W=torch.randn(7,9,dtype=dtype,generator=generator)*.4
        R=torch.randn(9,7,dtype=dtype,generator=generator)*.3
        bias=torch.randn(1,7,dtype=dtype,generator=generator)*.2
        target=torch.randn(1,7,dtype=dtype,generator=generator)
        def function(x):return x+(torch.tanh(x@W)@R if nonlinear else x@W@R)+bias
        norms=E.square().sum(-1)
        x,state=mixture(p,E,norms);output=function(x)
        def vjp(v):
            if nonlinear:return v+(v@R.T*(1-torch.tanh(x@W).square()))@W.T
            return v+(v@R.T)@W.T
        actual,info=gradient(output,target,E,norms,state,vjp)
        variable=p.detach().requires_grad_();mean=variable@E;energy=(variable*E.square().sum(-1)[None]).sum(-1,keepdim=True)
        independent=mean*energy.clamp_min(EPS*EPS).sqrt()/mean.square().sum(-1,keepdim=True).clamp_min(EPS*EPS).sqrt()
        result=function(independent)
        objective=1-torch.nn.functional.cosine_similarity(result,target,dim=-1).sum()
        expected=torch.autograd.grad(objective,variable)[0]
        atol=3e-5 if dtype==torch.float32 else 1e-10;rtol=3e-5 if dtype==torch.float32 else 1e-10
        torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
        torch.testing.assert_close(x,independent.detach(),rtol=rtol,atol=atol)
        relative=float((actual-expected).norm()/expected.norm().clamp_min(1e-30))
        cases.append({"dtype":str(dtype),"kind":kind,"nonlinear":nonlinear,"maximum_absolute_error":float((actual-expected).abs().max()),
          "relative_gradient_error":relative,"gradient_finite":bool(torch.isfinite(actual).all()),"passed":True})
      generator=torch.Generator().manual_seed(35002);E=torch.randn(23,7,dtype=dtype,generator=generator)
      norms=E.square().sum(-1);identity=torch.eye(len(E),dtype=dtype)
      points,state=mixture(identity,E,norms)
      torch.testing.assert_close(points,E,rtol=2e-6 if dtype==torch.float32 else 1e-12,atol=2e-6 if dtype==torch.float32 else 1e-12)
      p=torch.randn(5,23,dtype=dtype,generator=generator).softmax(-1);points,state=mixture(p,E,norms)
      torch.testing.assert_close(points.square().sum(-1,keepdim=True),state["energy"],rtol=2e-6 if dtype==torch.float32 else 1e-12,atol=2e-6 if dtype==torch.float32 else 1e-12)
      vertices.append({"dtype":str(dtype),"vertices":23,"one_hot_max_error":float((mixture(identity,E,norms)[0]-E).abs().max()),"energy_preserved":True,"passed":True})
    files=[Path(__file__),Path(__file__).with_name("normalized_mixture.py")]
    result={"task_id":"TRR-0020","passed":True,"gradient_cases":cases,"vertex_and_energy_checks":vertices,
      "scope":"independent tiny autograd; clamp-edge cases report relative errors because gradients can be large",
      "sources":{str(p.relative_to(ROOT)):sha(p) for p in files},"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"environment":{"python":sys.version,"torch":torch.__version__},"start_unix":start,"end_unix":time.time()}
    with (X/"dev35_cpu_reference.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({"passed":True,"gradient_cases":len(cases),"maximum_relative_gradient_error":max(v["relative_gradient_error"] for v in cases),"vertex_and_energy_checks":vertices},indent=2))
if __name__=="__main__":main()

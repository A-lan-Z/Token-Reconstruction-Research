"""Independent functional-attention dense derivatives and linear solves."""
from pathlib import Path
import sys,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage21"))
import torch
from torch.nn import functional as F
from homotopy import forward,jvp,global_gmres
from linearized_prefix import forward as original_forward
from linearized_prefix import rotate,split_heads,merge_heads
DEST=ROOT/"experiments/TRR-0020/dev39_cpu_reference.json"
def reference(x,layers,cos,sin,strength):
    h=x
    for p in layers:
        z=F.rms_norm(h,(h.shape[-1],),p["n1"],p["eps1"])
        q=rotate(split_heads(F.linear(z,p["q"]),p["heads"],p["head_dim"]),cos,sin)
        k=rotate(split_heads(F.linear(z,p["k"]),p["kv_heads"],p["head_dim"]),cos,sin).repeat_interleave(p["groups"],dim=0)
        v=split_heads(F.linear(z,p["v"]),p["kv_heads"],p["head_dim"]).repeat_interleave(p["groups"],dim=0)
        attention=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=p["scale"])
        middle=h+strength*F.linear(merge_heads(attention),p["o"])
        z=F.rms_norm(middle,(h.shape[-1],),p["n2"],p["eps2"])
        h=middle+strength*F.linear(F.silu(F.linear(z,p["gate"]))*F.linear(z,p["up"]),p["down"])
    return h
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    result={"task_id":"TRR-0020","scope":"tiny CPU homotopy algebra only; no real-prefix qualification or reconstruction",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv],
      "environment":{"torch":torch.__version__,"python":sys.version},"start_unix":time.time(),"passed":False,"rows":[],"linear":[]}
    paths=list(Path(__file__).parent.glob("*.py"))+[ROOT/"scripts/trr0020_stage21/linearized_prefix.py",ROOT/"experiments/TRR-0020/DEV39_PROSPECTIVE.md"]
    result["sources"]={str(p.relative_to(ROOT)):digest(p) for p in paths}
    try:
      for dtype in [torch.float32,torch.float64]:
       for length in [3,5]:
        generator=torch.Generator().manual_seed(39039+length)
        width=8;layers=[]
        for _ in range(2):
            p={key:.12*torch.randn(out,inp,generator=generator,dtype=dtype) for key,out,inp in
              [("q",8,8),("k",4,8),("v",4,8),("o",8,8),("gate",12,8),("up",12,8),("down",8,12)]}
            p.update(n1=1+.1*torch.randn(width,generator=generator,dtype=dtype),n2=1+.1*torch.randn(width,generator=generator,dtype=dtype),
              eps1=1e-5,eps2=1e-5,heads=2,kv_heads=1,head_dim=4,groups=2,scale=.5);layers.append(p)
        phase=torch.randn(length,2,generator=generator,dtype=dtype);phase=torch.cat([phase,phase],-1)
        co,si=phase.cos(),phase.sin();x=torch.randn(length,width,generator=generator,dtype=dtype)
        dx=torch.randn(length,width,generator=generator,dtype=dtype);dx[0].zero_()
        atol=2e-6 if dtype==torch.float32 else 2e-11;rtol=2e-5 if dtype==torch.float32 else 2e-10
        for strength in [0.,.25,.5,.75,1.]:
            lam=torch.tensor(strength,dtype=dtype)
            output,cache=forward(x,layers,co,si,lam)
            expected=reference(x,layers,co,si,lam)
            torch.testing.assert_close(output,expected,rtol=rtol,atol=atol)
            jx,jl=torch.autograd.functional.jacobian(lambda a,b:reference(a,layers,co,si,b),(x,lam))
            expected_input=torch.einsum("tdse,se->td",jx,dx)
            own_input=jvp(dx,layers,cache,co,si,lam)
            own_strength=jvp(torch.zeros_like(x),layers,cache,co,si,lam,1.)
            own_mixed=jvp(dx,layers,cache,co,si,lam,.37)
            torch.testing.assert_close(own_input,expected_input,rtol=rtol,atol=atol)
            torch.testing.assert_close(own_strength,jl,rtol=rtol,atol=atol)
            torch.testing.assert_close(own_mixed,expected_input+.37*jl,rtol=rtol,atol=atol)
            future=max(float(jx[i,:,j,:].abs().max()) for i in range(length) for j in range(i+1,length))
            past=max(float(jx[i,:,j,:].abs().max()) for i in range(length) for j in range(i))
            if future!=0 or not torch.equal(own_input[0],torch.zeros_like(own_input[0])):raise RuntimeError("causal/BOS input constraint")
            if strength==0 and not torch.equal(output,x):raise RuntimeError("identity endpoint")
            if strength==1:
                if not torch.equal(output,original_forward(x,layers,co,si)[0]):raise RuntimeError("original endpoint changed")
                if past<=1e-8:raise RuntimeError("coupled derivative was not exercised")
            result["rows"].append({"dtype":str(dtype),"length":length,"strength":strength,
              "forward_max_error":float((output-expected).abs().max()),"input_jvp_max_error":float((own_input-expected_input).abs().max()),
              "strength_jvp_max_error":float((own_strength-jl).abs().max()),
              "mixed_jvp_max_error":float((own_mixed-expected_input-.37*jl).abs().max()),
              "future_jacobian_max":future,"past_jacobian_max":past,"passed":True})
            if strength in [.5,1.]:
                matrix=jx[1:,:,1:,:].reshape((length-1)*width,(length-1)*width)
                b=torch.randn(length-1,width,generator=generator,dtype=dtype)
                def matvec(value):
                    padded=torch.cat([torch.zeros_like(value[:1]),value],0)
                    return jvp(padded,layers,cache,co,si,lam)[1:]
                ridge=1e-6 if dtype==torch.float32 else 1e-12
                actual,stats=global_gmres(matvec,b,len(b)*width,ridge)
                exact=torch.linalg.solve(matrix,b.flatten()).reshape_as(b)
                torch.testing.assert_close(actual,exact,rtol=2e-4 if dtype==torch.float32 else 2e-8,atol=2e-5 if dtype==torch.float32 else 2e-9)
                zero,_=global_gmres(matvec,torch.zeros_like(b),4,ridge)
                if not torch.equal(zero,torch.zeros_like(zero)):raise RuntimeError("zeroRHS")
                result["linear"].append({"dtype":str(dtype),"length":length,"strength":strength,"maximum_direction_error":float((actual-exact).abs().max()),
                  "relative_residual":float(stats["residual_norm"]/stats["rhs_norm"]),"matvec_calls":stats["matvec_calls"],"zero_rhs_exact":True,"passed":True})
      if len(result["rows"])!=20 or len(result["linear"])!=8:raise RuntimeError("incomplete CPUmatrix")
      result["passed"]=True
    except Exception:
      import traceback
      result["failure"]=traceback.format_exc();raise
    finally:
      result["end_unix"]=time.time()
      with DEST.open("x") as f:json.dump(result,f,indent=2)
      print(json.dumps(result,indent=2))
if __name__=="__main__":main()

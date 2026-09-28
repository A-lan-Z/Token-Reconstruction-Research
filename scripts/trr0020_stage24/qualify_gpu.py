"""Qualify normalized local operators against independent actual-prefix autograd."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020_stage21","scripts/trr0020_stage23"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
import torch
from torch.nn import functional as F
from linearized_prefix import forward,diagonal_jvp,public_parameters
from adjoint import diagonal_vjp
from cgls import least_squares
from normalized import equation
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev24_qualification"
DEST=X/"dev24_gpu_qualification.json"
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only qualification exists")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    cpu=json.loads((X/"dev24_cpu_reference_r1.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU reference failed")
    for path,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU source changed")
    paths=list((ROOT/"scripts/trr0020_stage24").glob("*.py"))
    paths +=[ROOT/p for p in ["scripts/trr0020_stage21/linearized_prefix.py","scripts/trr0020_stage23/adjoint.py",
      "scripts/trr0020_stage23/cgls.py","scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py"]]
    paths +=[X/p for p in ["DEV24_QUALIFICATION.md","dev24_preflight.json","dev24_cpu_reference_r1.json","dev21_gpu_qualification.json"]]
    q={"task_id":"TRR-0020","kind":"public_normalized_operator_qualification","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"contexts":[],"derivatives":[],"linear_checks":[],"passed":False}
    def save(name,data):
        p=OUT/(name+".safetensors");n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p)}
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);n.sync();q["prefix_preparation_seconds"]=time.perf_counter()-t
        old=json.loads((X/"dev21_gpu_qualification.json").read_text())
        for length in [128,40]:
            source=next(v for v in old["raw"] if v["path"].endswith(f"/{length}_forward.safetensors"))
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("changed public fixture")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda");x=fixture["input"];target=fixture["public_target"]
            positions=torch.arange(length,device="cuda")[None];pe=prefix.rotary_emb(x[None],positions);cos,sin=pe[0][0],pe[1][0]
            mask=prefix._causal_mask(x[None],start_pos=0,total_tokens=length)
            raw,cache=forward(x,layers,cos,sin)
            if not torch.equal(raw,fixture["actual"]):raise RuntimeError("changed FP32 forward")
            rhs,mv,rmv=equation(raw,target,lambda v:diagonal_jvp(v,layers,cache,cos,sin),lambda v:diagonal_vjp(v,layers,cache,cos,sin))
            rhs[0]=0
            def reference(value):
                hidden=value[None]
                for layer in prefix.layers:
                    hidden=prefix._hidden(layer(hidden,attention_mask=mask,position_ids=positions,use_cache=False,position_embeddings=pe))
                return F.normalize(hidden[0],dim=-1)
            expected=reference(x)
            actual=F.normalize(raw,dim=-1)
            context={"length":length,"source":source,"forward_equal":torch.equal(actual,expected),"minimum_prediction_norm":float(raw.norm(dim=-1).min()),
              **save(f"context_{length}",{"actual":actual,"reference":expected,"rhs":rhs})}
            q["contexts"].append(context)
            if not context["forward_equal"]:raise RuntimeError("normalized forward differs; saved")
            for pos in [1,length//2,length-1]:
                vector=torch.randn(2048,generator=torch.Generator().manual_seed(200058+length+pos)).to("cuda")/(2048**.5)
                tangent=torch.zeros_like(x);tangent[pos]=vector
                j=mv(tangent)[pos];jt=rmv(tangent)[pos]
                with torch.enable_grad():
                    _,reference_j=torch.autograd.functional.jvp(reference,x,tangent,create_graph=False,strict=True)
                    variable=x.detach().requires_grad_(True);value=reference(variable)
                    reference_jt=torch.autograd.grad(value,variable,grad_outputs=tangent)[0][pos].detach()
                reference_j=reference_j[pos].detach()
                row={"length":length,"position":pos,"jvp_max_error":float((j-reference_j).abs().max()),
                  "vjp_max_error":float((jt-reference_jt).abs().max()),
                  **save(f"derivatives_{length}_{pos}",{"tangent":vector,"jvp":j,"reference_jvp":reference_j,"vjp":jt,"reference_vjp":reference_jt})}
                q["derivatives"].append(row)
                torch.testing.assert_close(j,reference_j,rtol=5e-4,atol=5e-5)
                torch.testing.assert_close(jt,reference_jt,rtol=5e-4,atol=5e-5);row["passed"]=True
                del variable,value,reference_j,reference_jt,tangent,j,jt
                n.guard()
            for steps in [8,4]:
                for ridge in [.0001,.01]:
                    first=None;reps=[]
                    for rep in range(3):
                        n.sync();t=time.perf_counter();value,stats=least_squares(mv,rmv,rhs,steps,ridge);n.sync();seconds=time.perf_counter()-t
                        actual_residual=(mv(value)-rhs).norm(dim=-1)
                        data={"direction":value,"recurrence_residual":stats["recurrence_residual_norm"],"true_residual":actual_residual,"ridge":stats["ridge"]}
                        artifact=save(f"linear_{length}_{steps}_{ridge}_{rep}",data)
                        same=True if first is None else all(torch.equal(v,first[k]) for k,v in data.items())
                        if first is None:first={k:v.clone() for k,v in data.items()}
                        row={"rep":rep,"seconds":seconds,"repeat_equal":same,
                          "mean_relative_residual":float((actual_residual[1:]/rhs[1:].norm(dim=-1)).mean()),**artifact}
                        reps.append(row)
                        if not same or not torch.isfinite(value).all() or not torch.equal(value[0],torch.zeros_like(value[0])):raise RuntimeError("invalid/repeated linear direction; saved")
                        torch.testing.assert_close(stats["recurrence_residual_norm"],actual_residual,rtol=2e-4,atol=5e-6);n.guard()
                    q["linear_checks"].append({"length":length,"steps":steps,"ridge":ridge,"repetitions":reps,"jvp_calls_per_solve":steps,"vjp_calls_per_solve":steps+1,"validation_jvp_calls_per_solve":1})
            print("NORMALIZED_GEOMETRY_QUALIFIED",length,flush=True)
        if len(q["derivatives"])!=6 or len(q["linear_checks"])!=8:raise RuntimeError("incomplete qualification")
        q["passed"]=True
    except Exception:
        q["failure"]=traceback.format_exc();raise
    finally:
        n.sync();q.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,q);print("NORMALIZED_QUALIFICATION",q["passed"],flush=True)
if __name__=="__main__":main()

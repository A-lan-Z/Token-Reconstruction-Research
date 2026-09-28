"""Public qualification of full causal normalized inversion; never opens labels."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020_stage21","scripts/trr0020_stage23","scripts/trr0020_stage24"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
import torch
from torch.nn import functional as F
from linearized_prefix import forward,diagonal_jvp,public_parameters
from adjoint import diagonal_vjp
from cgls import least_squares as diagonal_solve
from global_cgls import least_squares as global_solve
from normalized import equation
from causal import operators,free_positions
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev25_qualification"
DEST=X/"dev25_gpu_qualification.json"

@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only qualification exists")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    cpu=json.loads((X/"dev25_cpu_reference.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU reference failed")
    for path,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU source changed")
    paths=list((ROOT/"scripts/trr0020_stage25").glob("*.py"))
    paths +=[ROOT/p for p in ["scripts/trr0020_stage21/linearized_prefix.py","scripts/trr0020_stage23/adjoint.py",
      "scripts/trr0020_stage23/cgls.py","scripts/trr0020_stage24/normalized.py",
      "scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py","scripts/trr0020_resource_guard.py"]]
    paths +=[X/p for p in ["DEV25_PROSPECTIVE.md","dev25_preflight.json","dev25_cpu_reference.json","dev21_gpu_qualification.json"]]
    q={"task_id":"TRR-0020","kind":"public_full_causal_operator_qualification","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"contexts":[],"derivatives":[],"linear_checks":[],"passed":False}
    def save(name,data):
        p=OUT/(name+".safetensors");n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p)}
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);radius=prefix.embed_tokens.weight.norm(dim=-1).median()
        n.sync();q["prefix_preparation_seconds"]=time.perf_counter()-t;q["direction_radius"]=float(radius)
        old=json.loads((X/"dev21_gpu_qualification.json").read_text())
        for length in [128,40]:
            source=next(v for v in old["raw"] if v["path"].endswith(f"/{length}_forward.safetensors"))
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("changed public fixture")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda");x=fixture["input"];target=fixture["public_target"]
            positions=torch.arange(length,device="cuda")[None];pe=prefix.rotary_emb(x[None],positions);cos,sin=pe[0][0],pe[1][0]
            mask=prefix._causal_mask(x[None],start_pos=0,total_tokens=length)
            raw,cache=forward(x,layers,cos,sin)
            if not torch.equal(raw,fixture["actual"]):raise RuntimeError("changed FP32 forward")
            raw_mv,raw_rmv=operators(layers,cache,cos,sin)
            rhs,mv,rmv=equation(raw,target,raw_mv,raw_rmv);rhs=free_positions(rhs)
            diag_rhs,diag_mv,diag_rmv=equation(raw,target,
              lambda v:diagonal_jvp(v,layers,cache,cos,sin),lambda v:diagonal_vjp(v,layers,cache,cos,sin))
            diag_rhs=free_positions(diag_rhs)
            if not torch.equal(rhs,diag_rhs):raise RuntimeError("changed comparison rhs")
            def reference_raw(value):
                hidden=value[None]
                for layer in prefix.layers:
                    hidden=prefix._hidden(layer(hidden,attention_mask=mask,position_ids=positions,use_cache=False,position_embeddings=pe))
                return hidden[0]
            expected=reference_raw(x)
            context={"length":length,"source":source,"forward_equal":torch.equal(raw,expected),
              **save(f"context_{length}",{"actual":raw,"reference":expected,"rhs":rhs})}
            q["contexts"].append(context)
            if not context["forward_equal"]:raise RuntimeError("forward differs; saved")
            for mode in ["raw","normalized"]:
                ref=reference_raw if mode=="raw" else lambda v:F.normalize(reference_raw(v),dim=-1)
                jfun,jtfun=(raw_mv,raw_rmv) if mode=="raw" else (mv,rmv)
                for index,kind in enumerate(["all","early","late"]):
                    vector=torch.randn(x.shape,generator=torch.Generator().manual_seed(200061+length+index)).to("cuda")/(2048**.5)
                    if kind!="all":
                        pos=1 if kind=="early" else length-1
                        keep=vector[pos].clone();vector.zero_();vector[pos]=keep
                    tangent=free_positions(vector)
                    j=jfun(vector);jt=jtfun(vector)
                    with torch.enable_grad():
                        _,reference_j=torch.autograd.functional.jvp(ref,x,tangent,create_graph=False,strict=True)
                        variable=x.detach().requires_grad_(True);value=ref(variable)
                        reference_jt=free_positions(torch.autograd.grad(value,variable,grad_outputs=vector)[0].detach())
                    reference_j=reference_j.detach()
                    row={"length":length,"mode":mode,"direction":kind,"jvp_max_error":float((j-reference_j).abs().max()),
                      "vjp_max_error":float((jt-reference_jt).abs().max()),
                      **save(f"derivatives_{length}_{mode}_{kind}",{"tangent":vector,"jvp":j,"reference_jvp":reference_j,"vjp":jt,"reference_vjp":reference_jt})}
                    q["derivatives"].append(row)
                    torch.testing.assert_close(j,reference_j,rtol=5e-4,atol=5e-5)
                    torch.testing.assert_close(jt,reference_jt,rtol=5e-4,atol=5e-5)
                    torch.testing.assert_close((vector*j).sum(),(jt*vector).sum(),rtol=2e-4,atol=5e-5)
                    row["passed"]=True
                    del variable,value,reference_j,reference_jt,tangent,j,jt
                    n.guard()
            target_unit=F.normalize(target,dim=-1)
            for steps in [16,8]:
                for ridge in [.0001,.01]:
                    for kind in ["full","diagonal"]:
                        solve=global_solve if kind=="full" else diagonal_solve
                        local_mv,local_rmv=(mv,rmv) if kind=="full" else (diag_mv,diag_rmv)
                        first=None;reps=[]
                        for rep in range(3):
                            n.sync();t=time.perf_counter()
                            direction,stats=solve(local_mv,local_rmv,rhs,steps,ridge)
                            n.sync();seconds=time.perf_counter()-t
                            equation_residual=local_mv(direction)-rhs
                            actual_residual=mv(direction)-rhs
                            factor=(radius/direction.norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
                            next_x=x+.5*factor[:,None]*direction;next_x[0]=x[0]
                            next_y,_=forward(next_x,layers,cos,sin)
                            nonlinear=free_positions(F.normalize(next_y,dim=-1)-target_unit)
                            data={"direction":direction,"recurrence_residual":stats["recurrence_residual_norm"],
                              "equation_residual":equation_residual,"full_linear_residual":actual_residual,
                              "nonlinear_residual":nonlinear,"ridge":stats["ridge"],"clip_factor":factor}
                            artifact=save(f"linear_{length}_{kind}_{steps}_{ridge}_{rep}",data)
                            same=True if first is None else all(torch.equal(v,first[k]) for k,v in data.items())
                            if first is None:first={k:v.clone() for k,v in data.items()}
                            row={"rep":rep,"seconds":seconds,"repeat_equal":same,
                              "equation_global_relative_residual":float(equation_residual.norm()/rhs.norm()),
                              "full_global_relative_residual":float(actual_residual.norm()/rhs.norm()),
                              "nonlinear_halfstep_global_relative_residual":float(nonlinear.norm()/rhs.norm()),
                              "clipped_fraction":float((factor[1:]<1).float().mean()),**artifact}
                            reps.append(row)
                            if not same or not all(torch.isfinite(v).all() for v in data.values()) or not torch.equal(direction[0],torch.zeros_like(direction[0])):
                                raise RuntimeError("invalid/repeated linear direction; saved")
                            torch.testing.assert_close(stats["recurrence_residual_norm"],equation_residual.norm(dim=-1),rtol=2e-4,atol=5e-6)
                            n.guard()
                        q["linear_checks"].append({"length":length,"kind":kind,"steps":steps,"ridge":ridge,"repetitions":reps,
                          "jvp_calls_per_solve":steps,"vjp_calls_per_solve":steps+1,"validation_jvp_calls_per_solve":2,"validation_full_forwards_per_solve":1})
            print("FULL_CAUSAL_GEOMETRY_QUALIFIED",length,flush=True)
        if len(q["derivatives"])!=12 or len(q["linear_checks"])!=16:raise RuntimeError("incomplete qualification")
        q["passed"]=True
    except Exception:
        q["failure"]=traceback.format_exc();raise
    finally:
        n.sync();q.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,q);print("FULL_CAUSAL_QUALIFICATION",q["passed"],flush=True)
if __name__=="__main__":main()

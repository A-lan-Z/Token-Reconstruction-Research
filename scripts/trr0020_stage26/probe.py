"""Fixed public conditioning matrix with exact unscaled controls."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020_stage21","scripts/trr0020_stage23","scripts/trr0020_stage24","scripts/trr0020_stage25"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
import torch
from torch.nn import functional as F
from linearized_prefix import forward,public_parameters
from global_cgls import least_squares
from normalized import equation
from causal import operators,free_positions
from precondition import make_scale,scaled_operators,KINDS
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev26_public"
DEST=X/"dev26_public_probe.json"
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only public probe exists")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    cpu=json.loads((X/"dev26_cpu_reference.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU scaling reference failed")
    for path,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU source changed")
    prior=json.loads((X/"dev25_gpu_qualification.json").read_text())
    if not prior["passed"]:raise RuntimeError("full causal operator unqualified")
    dependencies=["scripts/trr0020_stage21/linearized_prefix.py","scripts/trr0020_stage23/adjoint.py",
      "scripts/trr0020_stage25/global_cgls.py","scripts/trr0020_stage25/causal.py","scripts/trr0020_stage24/normalized.py",
      "scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py","scripts/trr0020_resource_guard.py"]
    for path in dependencies:
        if n.digest(ROOT/path)!=prior["sources"][path]:raise RuntimeError("qualified source changed")
    paths=list((ROOT/"scripts/trr0020_stage26").glob("*.py"))+[ROOT/p for p in dependencies]
    paths +=[X/p for p in ["DEV26_PROSPECTIVE.md","dev26_preflight.json","dev26_cpu_reference.json","dev25_gpu_qualification.json","dev21_gpu_qualification.json"]]
    result={"task_id":"TRR-0020","kind":"public_full_causal_conditioning","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"contexts":[],"cells":[],"passed":False,"control_anchors":0}
    def save(name,data):
        path=OUT/(name+".safetensors");start=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"save_and_hash_seconds":time.perf_counter()-start}
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);radius=prefix.embed_tokens.weight.norm(dim=-1).median()
        n.sync();result["prefix_preparation_seconds"]=time.perf_counter()-t;result["direction_radius"]=float(radius)
        old=json.loads((X/"dev21_gpu_qualification.json").read_text())
        for length in [128,40]:
            source=next(v for v in old["raw"] if v["path"].endswith(f"/{length}_forward.safetensors"))
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("changed public fixture")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda");x=fixture["input"];target=fixture["public_target"]
            positions=torch.arange(length,device="cuda")[None];pe=prefix.rotary_emb(x[None],positions);cos,sin=pe[0][0],pe[1][0]
            raw,cache=forward(x,layers,cos,sin)
            if not torch.equal(raw,fixture["actual"]):raise RuntimeError("changed forward")
            raw_mv,raw_rmv=operators(layers,cache,cos,sin)
            rhs,mv,rmv=equation(raw,target,raw_mv,raw_rmv);rhs=free_positions(rhs)
            generator=torch.Generator().manual_seed(200063+length)
            probes=[(torch.randint(0,2,x.shape,generator=generator)*2-1).to(device="cuda",dtype=x.dtype) for _ in range(4)]
            result["contexts"].append({"length":length,"source":source,"probe_seed":200063+length,
              **save(f"context_{length}",{"rhs":rhs,"probes":torch.stack(probes)})})
            target_unit=F.normalize(target,dim=-1)
            for steps in [32,16,8]:
                for ridge in [.0001,.01]:
                    for kind in ["coordinate_probe4","position_probe4","input_norm","identity"]:
                        first=None;reps=[]
                        for rep in range(3):
                            n.sync();t=time.perf_counter();scale,work=make_scale(kind,x,rmv,probes);n.sync();scale_seconds=time.perf_counter()-t
                            j,jt=scaled_operators(mv,rmv,scale)
                            t=time.perf_counter();value,stats=least_squares(j,jt,rhs,steps,ridge);direction=scale*value
                            n.sync();solve_seconds=time.perf_counter()-t
                            actual=mv(direction)-rhs
                            factor=(radius/direction.norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
                            next_x=x+.5*factor[:,None]*direction;next_x[0]=x[0]
                            next_y,_=forward(next_x,layers,cos,sin)
                            nonlinear=free_positions(F.normalize(next_y,dim=-1)-target_unit)
                            data={"direction":direction,"scale":scale,"recurrence_residual":stats["recurrence_residual_norm"],
                              "full_linear_residual":actual,"nonlinear_residual":nonlinear,"ridge":stats["ridge"],"clip_factor":factor}
                            artifact=save(f"{length}_{kind}_{steps}_{ridge}_{rep}",data)
                            same=True if first is None else all(torch.equal(v,first[k]) for k,v in data.items())
                            if first is None:first={k:v.clone() for k,v in data.items()}
                            row={"rep":rep,"scale_seconds":scale_seconds,"solve_seconds":solve_seconds,
                              "total_update_seconds":scale_seconds+solve_seconds,"repeat_equal":same,
                              "full_global_relative_residual":float(actual.norm()/rhs.norm()),
                              "nonlinear_halfstep_global_relative_residual":float(nonlinear.norm()/rhs.norm()),
                              "scale_min":float(scale[1:].min()),"scale_max":float(scale[1:].max()),
                              "clipped_fraction":float((factor[1:]<1).float().mean()),**artifact}
                            reps.append(row)
                            if not same or not all(torch.isfinite(v).all() for v in data.values()) or not torch.equal(direction[0],torch.zeros_like(direction[0])):
                                raise RuntimeError("invalid/repeated output; saved")
                            torch.testing.assert_close(stats["recurrence_residual_norm"],actual.norm(dim=-1),rtol=2e-4,atol=5e-6)
                            if kind=="identity" and steps in [8,16]:
                                anchor=next(v for v in prior["linear_checks"] if v["length"]==length and v["steps"]==steps and v["ridge"]==ridge and v["kind"]=="full")["repetitions"][rep]
                                if n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise RuntimeError("changed control archive")
                                original=n.load_file(str(ROOT/anchor["path"]),device="cuda")
                                keys=["direction","recurrence_residual","full_linear_residual","nonlinear_residual","ridge","clip_factor"]
                                equal=all(torch.equal(data[key],original[key]) for key in keys)
                                row["control_anchor"]={"path":anchor["path"],"sha256":anchor["sha256"],"equal":equal}
                                if not equal:raise RuntimeError("unscaled control changed; preserved")
                                result["control_anchors"]+=1
                            n.guard()
                        result["cells"].append({"length":length,"steps":steps,"ridge":ridge,"kind":kind,"repetitions":reps,
                          "jvp_calls_per_solve":steps,"vjp_calls_per_solve":steps+1,**work,
                          "validation_jvp_calls_per_solve":1,"validation_full_forwards_per_solve":1})
                    print("CONDITIONING_GROUP_COMPLETE",length,steps,ridge,flush=True)
        if len(result["cells"])!=48 or result["control_anchors"]!=24:raise RuntimeError("incomplete public matrix")
        result["passed"]=True
    except Exception:
        result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("CONDITIONING_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

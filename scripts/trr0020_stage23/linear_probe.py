"""Actual-prefix adjoint qualification and fixed public least-squares diagnostic."""
from pathlib import Path
import sys,os,json,time,hashlib,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020_stage21"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
import torch
from linearized_prefix import forward,diagonal_jvp,public_parameters
from krylov_graph import direction
from adjoint import diagonal_vjp
from cgls import least_squares
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev23_linear"
DEST=X/"dev23_linear_probe.json"
def tensor_hash(data):
    h=hashlib.sha256()
    for key in sorted(data):
        t=data[key].contiguous().cpu();h.update(key.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only diagnostic exists")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(200055)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    cpu=json.loads((X/"dev23_cpu_reference.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU qualification failed")
    for path,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU source changed")
    paths=list((ROOT/"scripts/trr0020_stage23").glob("*.py"))
    paths +=[ROOT/name for name in ["scripts/trr0020_stage21/krylov_graph.py","scripts/trr0020_stage21/linearized_prefix.py",
      "scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py"]]
    paths +=[X/name for name in ["DEV23_PROSPECTIVE.md","dev23_preflight.json","dev23_cpu_reference.json","dev21_gpu_qualification.json"]]
    q={"task_id":"TRR-0020","kind":"public_adjoint_and_linear_diagnostic","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":sys.argv,"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),"config_sha256":n.digest(n.ASSETS/"backup/config.json"),
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"adjoint_checks":[],"cases":[],"context":[],"passed":False}
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
            positions=torch.arange(length,device="cuda")[None]
            pe=prefix.rotary_emb(x[None],positions);cos,sin=pe[0][0],pe[1][0]
            mask=prefix._causal_mask(x[None],start_pos=0,total_tokens=length)
            prediction,cache=forward(x,layers,cos,sin)
            if not torch.equal(prediction,fixture["actual"]):raise RuntimeError("changed public forward")
            for pos in [1,length//2,length-1]:
                weights=torch.zeros_like(x);weights[pos]=torch.randn(2048,generator=torch.Generator().manual_seed(200055+length+pos)).to("cuda")/(2048**.5)
                actual=diagonal_vjp(weights,layers,cache,cos,sin)[pos]
                with torch.enable_grad():
                    variable=x.detach().requires_grad_(True);hidden=variable[None]
                    for layer in prefix.layers:
                        hidden=prefix._hidden(layer(hidden,attention_mask=mask,position_ids=positions,use_cache=False,position_embeddings=pe))
                    expected=torch.autograd.grad(hidden[0],variable,grad_outputs=weights)[0][pos].detach()
                check={"length":length,"position":pos,"max_absolute_error":float((actual-expected).abs().max()),
                  "relative_l2_error":float((actual-expected).norm()/expected.norm()),
                  **save(f"vjp_{length}_{pos}",{"actual":actual,"independent_HF_autograd":expected,"output_gradient":weights[pos]})}
                q["adjoint_checks"].append(check)
                torch.testing.assert_close(actual,expected,rtol=5e-4,atol=5e-5);check["passed"]=True
                del variable,hidden,weights,actual,expected
            perturbation=prefix.embed_tokens(fixture["public_ids"])-x;perturbation[0]=0
            consistent=diagonal_jvp(perturbation,layers,cache,cos,sin)
            observed=target-prediction;observed[0]=0
            q["context"].append({"length":length,"public_source":source,"forward_anchor_equal":True,
              **save(f"context_{length}",{"observed_rhs":observed,"consistent_rhs":consistent,"known_public_perturbation":perturbation})})
            n.guard()
            for rhs_name,rhs in [("observed_difference",observed),("consistent_linear",consistent)]:
                for steps in [16,4,8]:
                    for method,damping in [("krylov",None),("cgls0",0.),("cgls1e4",1e-4)]:
                        first=None;reps=[]
                        for rep in range(3):
                            mv=lambda v:diagonal_jvp(v,layers,cache,cos,sin)
                            rmv=lambda v:diagonal_vjp(v,layers,cache,cos,sin)
                            n.sync();t=time.perf_counter()
                            if method=="krylov":
                                value,reported,info=direction(mv,rhs,steps);ridge=torch.zeros(length,device="cuda")
                                jvp_calls=steps;vjp_calls=0
                            else:
                                value,stats=least_squares(mv,rmv,rhs,steps,damping)
                                reported=stats["recurrence_residual_norm"];ridge=stats["ridge"]
                                info=torch.zeros(length,device="cuda",dtype=torch.int32)
                                jvp_calls=stats["jvp_calls"];vjp_calls=stats["vjp_calls"]
                            n.sync();elapsed=time.perf_counter()-t
                            true=(mv(value)-rhs).norm(dim=-1)
                            tensors={"direction":value,"reported_residual":reported,"true_residual":true,"ridge":ridge,"solver_info":info}
                            sha=tensor_hash(tensors)
                            if first is None:
                                first=sha;artifact=save(f"{length}_{rhs_name}_{steps}_{method}",tensors)
                            elif sha!=first:
                                q["failure_artifact"]=save(f"FAILED_{length}_{rhs_name}_{steps}_{method}_{rep}",tensors)
                                raise RuntimeError("nonrepeatable direction; saved")
                            ratio=true[1:]/rhs[1:].norm(dim=-1).clamp_min(1e-20)
                            nonlinear={}
                            if rhs_name=="observed_difference":
                                for eta in [.5,1.]:
                                    changed,_=forward(x+eta*value,layers,cos,sin)
                                    nonlinear[str(eta)]=float((changed[1:]-target[1:]).norm()/observed[1:].norm())
                            reps.append({"direction_seconds":elapsed,"tensor_sha256":sha,"mean_relative_linear_residual":float(ratio.mean()),
                              "max_relative_linear_residual":float(ratio.max()),"mean_direction_norm":float(value[1:].norm(dim=-1).mean()),
                              "direction_error_to_public_perturbation":float((value[1:]-perturbation[1:]).norm(dim=-1).mean()) if rhs_name=="consistent_linear" else None,
                              "nonlinear_relative_residual_by_step_scale":nonlinear})
                            if not bool(torch.isfinite(value).all()) or not bool((info==0).all()):raise RuntimeError("invalid direction")
                            n.guard()
                        q["cases"].append({"length":length,"rhs":rhs_name,"steps":steps,"method":method,"damping":damping,
                          "jvp_calls_per_direction":jvp_calls,"vjp_calls_per_direction":vjp_calls,
                          "validation_jvp_calls_per_repetition":1,"nonlinear_forwards_per_repetition":2 if rhs_name=="observed_difference" else 0,
                          "repetitions":reps,"repeat_identical":True,**artifact})
                        print("DIRECTION",length,rhs_name,steps,method,round(reps[1]["mean_relative_linear_residual"],6),round(reps[1]["direction_seconds"],5),flush=True)
        if len(q["cases"])!=36 or len(q["adjoint_checks"])!=6:raise RuntimeError("incomplete public diagnostic")
        q["passed"]=True
    except Exception:
        q["failure"]=traceback.format_exc();raise
    finally:
        n.sync();q.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,q);print("ADJOINT_PROBE_RECEIPT",q["passed"],len(q["cases"]),flush=True)
if __name__=="__main__":main()

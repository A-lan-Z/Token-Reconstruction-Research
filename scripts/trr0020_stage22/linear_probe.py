"""Compare full-product and factorized Krylov solves on archived public fixtures."""
from pathlib import Path
import sys,os,time,json,hashlib,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020_stage21"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
import torch
from linearized_prefix import forward,diagonal_jvp,public_parameters
from krylov_graph import direction
from factorized import factorized_direction
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev22_linear"
DEST=X/"dev22_linear_probe.json"
def tensor_hash(data):
    h=hashlib.sha256()
    for key in sorted(data):
        t=data[key].contiguous().cpu();h.update(key.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only diagnostic exists")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(200053)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    cpu=json.loads((X/"dev22_cpu_reference.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU reference failed")
    for path,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU source changed")
    paths=[Path(__file__),ROOT/"scripts/trr0020_stage22/factorized.py",ROOT/"scripts/trr0020_stage21/krylov_graph.py",
      ROOT/"scripts/trr0020_stage21/linearized_prefix.py",ROOT/"scripts/trr0014/native.py",ROOT/"scripts/agent4/common.py",
      ROOT/"src/token_reconstruction/public_prefix.py",X/"DEV22_PROSPECTIVE.md",X/"dev22_preflight.json",X/"dev22_cpu_reference.json",X/"dev21_gpu_qualification.json"]
    q={"task_id":"TRR-0020","kind":"public_linear_diagnostic_only","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"command":sys.argv,"start_unix":time.time(),
      "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),"config_sha256":n.digest(n.ASSETS/"backup/config.json"),
      "truth_read":False,"shortlist":None,"cases":[],"context":[],"passed":False}
    def save(name,data):
        path=OUT/(name+".safetensors")
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path)}
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");layers=public_parameters(prefix);n.sync()
        q["prefix_preparation_seconds"]=time.perf_counter()-t
        old=json.loads((X/"dev21_gpu_qualification.json").read_text())
        for length in [128,40]:
            source=next(e for e in old["raw"] if e["path"].endswith(f"/{length}_forward.safetensors"))
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("changed public fixture")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda");x=fixture["input"];target=fixture["public_target"]
            cos,sin=prefix.rotary_emb(x[None],torch.arange(length,device="cuda")[None]);cos=cos[0];sin=sin[0]
            prediction,cache=forward(x,layers,cos,sin)
            if not torch.equal(prediction,fixture["actual"]):raise RuntimeError("public forward anchor changed")
            perturbation=prefix.embed_tokens(fixture["public_ids"])-x;perturbation[0]=0
            consistent=diagonal_jvp(perturbation,layers,cache,cos,sin)
            composed=perturbation
            for p,c in zip(layers,cache):composed=diagonal_jvp(composed,[p],[c],cos,sin)
            observed=target-prediction;observed[0]=0
            context={"length":length,"public_source":source,"forward_anchor_equal":True,"factor_composition_equal":torch.equal(consistent,composed),
              **save(f"context_{length}",{"observed_rhs":observed,"consistent_rhs":consistent,"known_public_perturbation":perturbation,"composed_product":composed})}
            q["context"].append(context)
            if not context["factor_composition_equal"]:raise RuntimeError("Jacobian factor composition failed; saved")
            for rhs_name,rhs in [("observed_difference",observed),("consistent_linear",consistent)]:
                for k in [16,4,8]:
                    for method in ["full_product","layer_factors"]:
                        first=None;repetitions=[]
                        for rep in range(3):
                            n.sync();t=time.perf_counter()
                            if method=="full_product":
                                value,aux,info=direction(lambda v:diagonal_jvp(v,layers,cache,cos,sin),rhs,k)
                                aux=aux[None]
                            else:value,aux,info=factorized_direction(layers,cache,cos,sin,rhs,k)
                            n.sync();elapsed=time.perf_counter()-t
                            true_residual=(diagonal_jvp(value,layers,cache,cos,sin)-rhs).norm(dim=-1)
                            tensors={"direction":value,"auxiliary_residual":aux,"true_residual":true_residual,"solver_info":info}
                            signature=tensor_hash(tensors)
                            if first is None:
                                artifact=save(f"{length}_{rhs_name}_k{k}_{method}",tensors);first=signature
                            elif signature!=first:
                                artifact_failure=save(f"FAILED_{length}_{rhs_name}_k{k}_{method}_rep{rep}",tensors)
                                q["failure_artifact"]=artifact_failure
                                raise RuntimeError("nonrepeatable linear probe; saved")
                            ratio=true_residual[1:]/rhs[1:].norm(dim=-1).clamp_min(1e-20)
                            repetitions.append({"seconds_direction_only":elapsed,"tensor_sha256":signature,
                              "mean_relative_residual":float(ratio.mean()),"max_relative_residual":float(ratio.max()),
                              "mean_direction_norm":float(value[1:].norm(dim=-1).mean()),
                              "mean_direction_error_to_public_perturbation":float((value[1:]-perturbation[1:]).norm(dim=-1).mean()) if rhs_name=="consistent_linear" else None})
                            if not bool(torch.isfinite(value).all()) or not bool((info==0).all()):raise RuntimeError("invalid linear solve")
                            n.guard()
                        case={"length":length,"rhs":rhs_name,"basis_vectors":k,"method":method,"repetitions":repetitions,**artifact,
                          "single_layer_products_per_direction":4*k,"small_linear_solves_per_direction":1 if method=="full_product" else 4,
                          "additional_full_prefix_products_per_repetition_for_residual_validation":1,"repeat_identical":True}
                        q["cases"].append(case)
                        print("LINEAR",length,rhs_name,k,method,round(repetitions[1]["mean_relative_residual"],6),round(repetitions[1]["seconds_direction_only"],5),flush=True)
            n.guard()
        if len(q["cases"])!=24:raise RuntimeError("incomplete public diagnostic")
        q["passed"]=True
    except Exception:
        q["failure"]=traceback.format_exc()
        raise
    finally:
        n.sync();q.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,q)
        print("LINEAR_PROBE_RECEIPT",q["passed"],len(q["cases"]),flush=True)
if __name__=="__main__":main()

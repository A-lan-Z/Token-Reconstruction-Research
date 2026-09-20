"""Public-only loss curves toward known fixture vertices; not a decoder."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource,importlib.util
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage35"))
spec=importlib.util.spec_from_file_location("qualified_norm_probe",ROOT/"scripts/trr0020_stage35/probe.py")
qualified=importlib.util.module_from_spec(spec);sys.modules[spec.name]=qualified;spec.loader.exec_module(qualified)
n=qualified.n;torch=qualified.torch;F=qualified.F;forward=qualified.forward;commit=qualified.commit
public_parameters=qualified.public_parameters;guard=qualified.guard
from single_position import jvp
from normalized_mixture import mixture,EPS
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev36_public";DEST=X/"dev36_public_curve.json"
ALPHAS=[0.,.0001,.001,.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,1.]
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only public diagnostic")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism flags")
    raw=json.loads((X/"dev34_stress_probe.json").read_text());normal=json.loads((X/"dev35_public_probe.json").read_text())
    if not raw["passed"] or not normal["passed"]:raise RuntimeError("unqualified source studies")
    for q in [raw,normal]:
        for path,sha in q["sources"].items():
            if n.digest(ROOT/path)!=sha:raise RuntimeError("source changed "+path)
    paths=list(Path(__file__).parent.glob("*.py"))+[X/name for name in ["DEV36_PLAN.md","dev36_preflight.json","dev34_stress_probe.json","dev35_public_probe.json"]]
    result={"task_id":"TRR-0020","scope":"public known-vertex loss-landscape diagnostic; identities guide diagnostic paths only, never reconstruction",
      "sources":normal["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"alphas":ALPHAS,
      "prefix_sha256":normal["prefix_sha256"],"config_sha256":normal["config_sha256"],
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "environment":n.environment(),"start_unix":time.time(),"passed":False,"truth_read":False,"curves":[],"input_anchors":[]}
    def save(name,data):
        p=OUT/(name+".safetensors");t=time.perf_counter();n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard()
        fixture=raw["fixture"]
        if fixture!=normal["fixture"] or n.digest(ROOT/fixture["path"])!=fixture["sha256"]:raise RuntimeError("fixture changed")
        data=n.load_file(str(ROOT/fixture["path"]));ids=data["public_ids"].to("cuda")
        if n.digest(n.ASSETS/"backup/prefix.safetensors")!=normal["prefix_sha256"]:raise RuntimeError("prefix changed")
        n.sync();t=time.perf_counter();prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        E=prefix.embed_tokens.weight.detach();layers=public_parameters(prefix);norms=E.square().sum(-1)
        n.sync();result["preparation_seconds"]=time.perf_counter()-t
        base=prefix.embed_tokens(ids[0]);rotary=prefix.rotary_emb(base[None],torch.arange(2,device="cuda")[None]);co,si=rotary[0][0],rotary[1][0]
        empty=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in layers]
        _,_,bos=forward(E[128000:128001],layers,empty,co[0:1],si[0:1]);past=commit(empty,bos)
        immutable=[(k.clone(),v.clone()) for k,v in past]
        for precision,key in [("fp32","fp32_target"),("bf16","bf16_target")]:
          for index,text in enumerate(raw["public_texts"]):
            target=data[key][index,1:2].to(device="cuda",dtype=torch.float32);actual_id=int(ids[index,1])
            expected_point=E[actual_id:actual_id+1]
            exact,_,_=forward(expected_point,layers,past,co[1:2],si[1:2])
            fp32_reference=data["fp32_target"][index,1:2].to("cuda")
            full_error=float((exact-fp32_reference).abs().max())
            torch.testing.assert_close(exact,fp32_reference,rtol=5e-4,atol=5e-5)
            result["input_anchors"].append({"precision":precision,"context":index,"native_vs_full_max_error":full_error,
              "cosine_error_to_observation":float(1-F.cosine_similarity(exact,target).sum()),"passed":True})
            for mode,q in [("ordinary",raw),("normalized",normal)]:
                ref=next(v for v in q["gradient_references"] if v["precision"]==precision and v["context"]==index)
                if n.digest(ROOT/ref["path"])!=ref["sha256"]:raise RuntimeError("reference changed")
                initial=n.load_file(str(ROOT/ref["path"]),device="cuda");z=initial["logits"];p=z.softmax(-1)
                raw_mean=p@E
                if mode=="ordinary":
                    point=raw_mean;tangent=expected_point-raw_mean
                else:
                    point,state=mixture(p,E,norms)
                    mean_delta=expected_point-state["mean"];energy_delta=norms[actual_id]-state["energy"]
                    ratio=.5*(energy_delta/state["energy"].clamp_min(EPS*EPS)*(state["energy"]>EPS*EPS)
                      -2*(state["mean"]*mean_delta).sum(-1,keepdim=True)/state["raw_squared"].clamp_min(EPS*EPS)*(state["raw_squared"]>EPS*EPS))
                    tangent=state["scale"]*(mean_delta+state["mean"]*ratio)
                initial_out,cache,_=forward(point,layers,past,co[1:2],si[1:2])
                if not torch.equal(initial_out,initial["output"]):raise RuntimeError("initial output anchor changed")
                dy=jvp(tangent,layers,cache,co[1:2],si[1:2])
                unit=F.normalize(initial_out,dim=-1);unit_target=F.normalize(target,dim=-1)
                cotangent=(-unit_target+unit*(unit*unit_target).sum(-1,keepdim=True))/initial_out.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                slope_jvp=(cotangent*dy).sum()
                slope_gradient=initial["gradient"][0,actual_id]-(initial["gradient"]*p).sum()
                torch.testing.assert_close(slope_jvp,slope_gradient,rtol=5e-4,atol=5e-5)
                mse_slope=2*((initial_out-target)*dy).sum()/target.square().sum().clamp_min(1e-24)
                inputs=[];outputs=[];losses=[];mse=[]
                n.sync();t=time.perf_counter()
                for alpha in ALPHAS:
                    distribution=(1-alpha)*p;distribution[0,actual_id]+=alpha
                    if mode=="ordinary":point=distribution@E
                    else:point,_=mixture(distribution,E,norms)
                    out,_,_=forward(point,layers,past,co[1:2],si[1:2])
                    inputs.append(point.clone());outputs.append(out.clone())
                    losses.append(1-F.cosine_similarity(out,target,dim=-1))
                    mse.append((out-target).square().sum(-1)/target.square().sum(-1).clamp_min(1e-24))
                n.sync();seconds=time.perf_counter()-t
                torch.testing.assert_close(inputs[-1],expected_point,rtol=2e-6,atol=2e-7)
                torch.testing.assert_close(outputs[-1],fp32_reference,rtol=5e-4,atol=5e-5)
                arrays={"alphas":torch.tensor(ALPHAS),"inputs":torch.cat(inputs),"outputs":torch.cat(outputs),
                  "cosine_error":torch.cat(losses),"normalized_squared_error":torch.cat(mse),"observation":target,
                  "native_vertex_output":exact,"full_fp32_reference":fp32_reference,"initial_direction":tangent,"initial_jvp":dy}
                if not all(bool(torch.isfinite(v).all()) for v in arrays.values()):raise RuntimeError("nonfinite curve")
                true_logit=z[0,actual_id];true_gradient=initial["gradient"][0,actual_id]
                row={"precision":precision,"context":index,"public_text":text,"mode":mode,"reference":ref,
                  "initial_true_probability":float(p[0,actual_id]),"initial_true_logit_rank":int((z[0]>true_logit).sum())+1,
                  "initial_true_gradient_rank":int((initial["gradient"][0]<true_gradient).sum())+1,
                  "initial_cosine_slope":float(slope_gradient),"jvp_cosine_slope":float(slope_jvp),"initial_squared_error_slope":float(mse_slope),
                  "cosine_errors":[float(v) for v in arrays["cosine_error"]],"normalized_squared_errors":[float(v) for v in arrays["normalized_squared_error"]],
                  "curve_seconds":seconds,"prefix_forwards":len(ALPHAS)+1,"prefix_jvps":1,"passed":True,
                  **save(f"{precision}_{index}_{mode}",arrays)}
                result["curves"].append(row)
                guard()
            if not all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(past,immutable)):raise RuntimeError("history changed")
            print("PUBLIC_VERTEX_CURVES_COMPLETE",precision,index,flush=True)
        if len(result["curves"])!=64 or len(result["input_anchors"])!=32:raise RuntimeError("incomplete curve matrix")
        for path,sha in result["sources"].items():
            if n.digest(ROOT/path)!=sha:raise RuntimeError("source changed during run")
        result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("PUBLIC_VERTEX_CURVES_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

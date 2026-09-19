"""Public-only comparison of fixed current-token input metrics."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage27"))
from qualify_gpu import n,torch,F,public_parameters,forward,jvp,vjp,commit,equation
from metric_direction import build,direction,METRICS,RULES
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev28_public";DEST=X/"dev28_public_probe.json"

@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only metric probe")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("deterministic environment")
    cpu=json.loads((X/"dev28_cpu_reference.json").read_text())
    old=json.loads((X/"dev27_cheap_probe.json").read_text())
    if not cpu["result"]["passed"] or not old["passed"]:raise RuntimeError("unqualified source")
    for record,field in [(cpu,"source_sha256"),(old,"sources")]:
        for name,sha in record[field].items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("qualified source changed "+name)
    paths=list((ROOT/"scripts/trr0020_stage28").glob("*.py"))
    paths +=[ROOT/v for v in old["sources"]]
    paths +=[ROOT/"src/token_reconstruction/prefix_weight_metric.py"]
    paths +=[X/v for v in ["DEV28_PROSPECTIVE.md","DEV28_EXECUTION.md","dev28_preflight.json","dev28_cpu_reference.json","dev27_cheap_probe.json"]]
    result={"task_id":"TRR-0020","kind":"public_current_token_metric_directions","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "sources":{str(v.relative_to(ROOT)):n.digest(v) for v in paths},"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"contexts":[],"cells":[],"passed":False,"forward_anchors":0,"identity_direction_anchors":0}
    def save(name,data):
        path=OUT/(name+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"save_and_hash_seconds":time.perf_counter()-t}
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);radius=prefix.embed_tokens.weight.norm(dim=-1).median()
        n.sync();result["prefix_preparation_seconds"]=time.perf_counter()-t
        t=time.perf_counter();metric=PrefixWeightMetric(prefix);result["lookup_preparation_stats"]=metric.build()
        n.sync();result["lookup_preparation_seconds"]=time.perf_counter()-t
        t=time.perf_counter();ps,raw=build(metric.transform);n.sync();result["preconditioner_seconds"]=time.perf_counter()-t
        result["preconditioners"]=save("preconditioners",{"transform":metric.transform,"A":raw["matrix"],"inverseA":raw["inverse"],**{k:v for k,v in ps.items() if v is not None}})
        result["regularizer"]=raw["regularizer"];result["metric_diagnostics"]=[]
        validation=time.perf_counter()
        for name,p in ps.items():
            if p is None:continue
            eig=torch.linalg.eigvalsh(p.cpu().double())
            if not (eig>0).all():raise RuntimeError("real-prefix metric not SPD")
            result["metric_diagnostics"].append({"metric":name,"min_eigenvalue":float(eig.min()),"max_eigenvalue":float(eig.max()),
              "condition":float(eig.max()/eig.min()),"max_asymmetry":float((p-p.T).abs().max()),"bytes":p.numel()*p.element_size()})
        inverse_error=(raw["matrix"]@raw["inverse"]-torch.eye(2048,device="cuda")).abs().max()
        result["inverse_max_absolute_residual"]=float(inverse_error)
        if not torch.isfinite(inverse_error) or inverse_error>5e-3:raise RuntimeError("inverse residual qualification")
        n.sync();result["metric_validation_seconds"]=time.perf_counter()-validation;n.guard()
        for length in [128,40]:
            context=next(v for v in old["contexts"] if v["length"]==length);source=context["source"]
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("public fixture changed")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda")
            base=prefix.embed_tokens(fixture["public_ids"]);target=fixture["public_target"];noisy=fixture["input"]
            pe=prefix.rotary_emb(base[None],torch.arange(length,device="cuda")[None]);cos,sin=pe[0][0],pe[1][0]
            selected=[length-1,length//2,1];snapshots={}
            past=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in layers]
            t=time.perf_counter()
            for pos in range(length):
                if pos in selected:snapshots[pos]=past
                _,_,current=forward(base[pos:pos+1],layers,past,cos[pos:pos+1],sin[pos:pos+1]);past=commit(past,current)
            n.sync();cache_seconds=time.perf_counter()-t
            for pos in selected:
                cur=noisy[pos:pos+1];history=snapshots[pos];co,si=cos[pos:pos+1],sin[pos:pos+1]
                n.sync();t=time.perf_counter();prediction,caches,_=forward(cur,layers,history,co,si);n.sync();forward_seconds=time.perf_counter()-t
                context=next(v for v in old["contexts"] if v["length"]==length and v["position"]==pos);anchor=context["forward_anchor"]
                if n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise RuntimeError("forward anchor changed")
                original=n.load_file(str(ROOT/anchor["path"]),device="cuda")
                equal=torch.equal(prediction,original["actual"]) and torch.equal(cur,original["input"])
                result["contexts"].append({"length":length,"position":pos,"source":source,"forward_anchor":anchor,"equal":equal,
                  "forward_seconds":forward_seconds,"public_cache_commit_seconds":cache_seconds})
                if not equal:raise RuntimeError("current map changed")
                result["forward_anchors"]+=1
                rhs,mv,rmv=equation(prediction,target[pos:pos+1],lambda v:jvp(v,layers,caches,co,si),lambda v:vjp(v,layers,caches,co,si))
                for name,p in ps.items():
                    for rule in RULES:
                        first=None;reps=[]
                        control=next(v for v in old["cells"] if v["length"]==length and v["position"]==pos and v["rule"]==rule)
                        reference=control["repetitions"][0]
                        if n.digest(ROOT/reference["path"])!=reference["sha256"]:raise RuntimeError("direction anchor changed")
                        archived=n.load_file(str(ROOT/reference["path"]),device="cuda")
                        for rep in range(3):
                            n.sync();t=time.perf_counter();delta,work=direction(rule,rhs,mv,rmv,p);n.sync();seconds=time.perf_counter()-t
                            factor=(radius/delta.norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
                            t=time.perf_counter()
                            one,_,_=forward(cur+factor[:,None]*delta,layers,history,co,si)
                            half,_,_=forward(cur+.5*factor[:,None]*delta,layers,history,co,si)
                            target_unit=F.normalize(target[pos:pos+1],dim=-1)
                            one_error=F.normalize(one,dim=-1)-target_unit;half_error=F.normalize(half,dim=-1)-target_unit
                            data={"direction":delta,"linear_residual":mv(delta)-rhs,"one_step_residual":one_error,"half_step_residual":half_error,"clip_factor":factor}
                            n.sync();validation_seconds=time.perf_counter()-t
                            artifact=save(f"{length}_{pos}_{name}_{rule}_{rep}",data)
                            repeat=True if first is None else all(torch.equal(v,first[k]) for k,v in data.items())
                            if first is None:first={k:v.clone() for k,v in data.items()}
                            row={"rep":rep,"seconds":seconds,"validation_seconds":validation_seconds,"repeat_equal":repeat,
                              "linear_relative_residual":float(data["linear_residual"].norm()/rhs.norm()),
                              "one_step_relative_residual":float(one_error.norm()/rhs.norm()),
                              "half_step_relative_residual":float(half_error.norm()/rhs.norm()),**artifact}
                            reps.append(row)
                            if not repeat or not all(torch.isfinite(v).all() for v in data.values()):raise RuntimeError("invalid output; preserved")
                            if name=="identity":
                                row["archived_exact"]=all(torch.equal(v,archived[k]) for k,v in data.items())
                                if not row["archived_exact"]:raise RuntimeError("identity output changed")
                                result["identity_direction_anchors"]+=1
                            n.guard()
                        result["cells"].append({"length":length,"position":pos,"metric":name,"rule":rule,"repetitions":reps,**work,
                          "validation_jvp_calls_per_solve":1,"validation_forward_calls_per_solve":2})
                print("METRIC_CONTEXT_COMPLETE",length,pos,flush=True)
        if len(result["cells"])!=36 or result["forward_anchors"]!=6 or result["identity_direction_anchors"]!=36:raise RuntimeError("incomplete matrix")
        for name,sha in result["sources"].items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed during run")
        result["passed"]=True
    except Exception:
        result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("METRIC_CURRENT_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

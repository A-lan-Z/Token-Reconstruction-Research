"""Public current-token full-vocabulary updates with immutable synthetic history."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage27"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage32"))
from qualify_gpu import n,torch,F,public_parameters,forward,jvp,vjp,commit,FrozenPast
from probability_gradient import gradient as probability_gradient
from budget_step import update
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev33_public";DEST=X/"dev33_public_probe.json"
PHASES=["probability_and_mixture","current_prefix_forward","cosine_cotangent","current_prefix_vjp","full_vocabulary_gradient","probability_update"]
def guard():
    free,_=torch.cuda.mem_get_info()
    available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
    temperature=int(subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],text=True).strip())
    if free<3*2**30 or available<8*2**30 or torch.cuda.max_memory_reserved()>8*2**30 or temperature>=80:raise RuntimeError("resource margin violated")

@torch.no_grad()
def trajectory(z0,target,E,layers,past,co,si,factor,steps):
    z=z0.clone();records=[];phase_events=[];logits=[z.clone()];budgets=[]
    n.sync();start=time.perf_counter()
    for _ in range(steps):
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(7)]
        ev[0].record()
        p=z.softmax(-1);mean=p@E;ev[1].record()
        out,caches,_=forward(mean,layers,past,co,si);ev[2].record()
        def reverse(cotangent):
            ev[3].record();value=vjp(cotangent,layers,caches,co,si);ev[4].record();return value
        G,info=probability_gradient(out,target,E,reverse);ev[5].record()
        z,step=update(z,G,info["loss"],factor,4);ev[6].record()
        records.append(torch.stack([info["loss"][0],p.amax(),step["requested_budget"][0],step["formula_kl"][0],step["budget_used"][0]]))
        phase_events.append(ev);logits.append(z.clone());budgets.append(step["requested_budget"].clone())
    probability=z.softmax(-1);mean=probability@E
    out,_,_=forward(mean,layers,past,co,si)
    error=1-(F.normalize(out,dim=-1)*F.normalize(target,dim=-1)).sum(-1)
    token=z.argmax(-1)
    n.sync();inference=time.perf_counter()-start
    start=time.perf_counter()
    commit_output,_,current=forward(E.index_select(0,token),layers,past,co,si)
    committed=commit(past,current);n.sync();commit_seconds=time.perf_counter()-start
    phases={name:sum(ev[i].elapsed_time(ev[i+1])/1000 for ev in phase_events) for i,name in enumerate(PHASES)}
    validate=time.perf_counter();actual=[]
    for old,new in zip(logits[:-1],logits[1:]):
        a=old.cpu().double().log_softmax(-1);b=new.cpu().double().log_softmax(-1)
        actual.append((a.exp()*(a-b)).sum(-1))
    direct=torch.stack(actual)[:,0];budget=torch.stack(budgets).cpu()[:,0].double()
    data={"logits":z,"mixture":mean,"prefix_output":out,"observed_error":error,"confidence":probability.amax(-1),
      "token":token,"trace":torch.stack(records),"direct_kl":direct,"requested_budget":budget,"commit_output":commit_output}
    for i,(key,value) in enumerate(current):data[f"commit_key_{i}"]=key;data[f"commit_value_{i}"]=value
    numeric=all(bool(torch.isfinite(v).all()) for v in data.values())
    valid=numeric and bool((direct<=budget+2e-5).all()) and bool((direct>=-2e-5).all())
    n.sync()
    return data,{"inference_seconds":inference,"emitted_token_commit_seconds":commit_seconds,
      "phase_gpu_seconds":phases,"validation_seconds":time.perf_counter()-validate,"numeric_valid":numeric,"passed":valid,
      "prefix_forwards":steps+1,"prefix_vjps":steps,"vocabulary_mixture_products":steps+1,"vocabulary_gradient_products":steps,
      "commit_prefix_forwards":1,"probability_update_iterations":4,"vocabulary":len(E),"shortlist":None,
      "model_parameter_updates":0,"timing_scope":"eager current-token updates and final mixture evaluation; per-step diagnostic clones included; token commit and FP64 validation separate"}

@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only public probe")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("deterministic environment")
    cpu=json.loads((X/"dev33_cpu_reference.json").read_text())
    old=json.loads((X/"dev27_gpu_qualification.json").read_text())
    budget_cpu=json.loads((X/"dev32_cpu_reference.json").read_text())
    if not cpu["result"]["passed"] or not budget_cpu["result"]["passed"] or not old["passed"]:raise RuntimeError("unqualified primitives")
    for record in [cpu,budget_cpu]:
        for name,sha in record["sources"].items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("qualified source changed")
    dependencies=["scripts/trr0020_stage27/single_position.py","scripts/trr0020_stage21/linearized_prefix.py",
      "scripts/trr0020_stage23/adjoint.py","scripts/trr0020_stage24/normalized.py","scripts/trr0014/native.py",
      "scripts/agent4/common.py","src/token_reconstruction/public_prefix.py","scripts/trr0020_resource_guard.py"]
    for name in dependencies:
        if n.digest(ROOT/name)!=old["sources"][name]:raise RuntimeError("prefix derivative source changed")
    paths=[ROOT/name for name in dependencies]+list((ROOT/"scripts/trr0020_stage33").glob("*.py"))
    paths +=[ROOT/name for name in ["scripts/trr0020_stage27/qualify_gpu.py","scripts/trr0020_stage32/budget_step.py","src/token_reconstruction/prefix_weight_metric.py"]]
    paths +=[X/name for name in ["DEV33_PROSPECTIVE.md","DEV33_PUBLIC_PLAN.md","dev33_preflight.json","dev33_cpu_reference.json","dev32_cpu_reference.json","dev27_gpu_qualification.json"]]
    result={"task_id":"TRR-0020","kind":"public_causal_full_vocabulary_probability_updates",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":sys.argv,"start_unix":time.time(),"passed":False,
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),"config_sha256":n.digest(n.ASSETS/"backup/config.json"),
      "truth_read":False,"shortlist":None,"scope":"public synthetic fixed past; not reconstruction accuracy",
      "contexts":[],"cells":[],"forward_anchors":0,"derivative_anchors":0,"probability_gradient_references":[]}
    def save(name,data):
        path=OUT/(name+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);E=prefix.embed_tokens.weight.detach()
        n.sync();result["prefix_preparation_seconds"]=time.perf_counter()-t
        t=time.perf_counter();metric=PrefixWeightMetric(prefix);result["metric_stats"]=metric.build();n.sync()
        result["metric_preparation_seconds"]=time.perf_counter()-t
        for length in [128,40]:
            source=next(v for v in old["contexts"] if v["length"]==length)["source"]
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("public source changed")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda")
            base=prefix.embed_tokens(fixture["public_ids"]);target=fixture["public_target"];noisy=fixture["input"]
            pe=prefix.rotary_emb(base[None],torch.arange(length,device="cuda")[None]);cos,sin=pe[0][0],pe[1][0]
            if not torch.equal(prefix.forward_full(fixture["public_ids"][None])[0],target):raise RuntimeError("target anchor changed")
            selected=[length-1,length//2,1];snapshots={}
            past=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in layers]
            n.sync();t=time.perf_counter()
            for pos in range(length):
                if pos in selected:snapshots[pos]=past
                _,_,current=forward(base[pos:pos+1],layers,past,cos[pos:pos+1],sin[pos:pos+1]);past=commit(past,current)
            n.sync();public_history_seconds=time.perf_counter()-t
            for pos in selected:
                history=snapshots[pos];unchanged=[(k.clone(),v.clone()) for k,v in history]
                co,si=cos[pos:pos+1],sin[pos:pos+1];truthless=target[pos:pos+1]
                old_forward=next(v for v in old["forwards"] if v["length"]==length and v["position"]==pos)
                if n.digest(ROOT/old_forward["path"])!=old_forward["sha256"]:raise RuntimeError("forward artifact changed")
                original=n.load_file(str(ROOT/old_forward["path"]),device="cuda")
                raw,caches,_=forward(noisy[pos:pos+1],layers,history,co,si)
                if not torch.equal(raw,original["actual"]):raise RuntimeError("forward implementation changed")
                result["forward_anchors"]+=1
                der=next(v for v in old["derivatives"] if v["length"]==length and v["position"]==pos and v["mode"]=="raw")
                if n.digest(ROOT/der["path"])!=der["sha256"]:raise RuntimeError("derivative artifact changed")
                original_d=n.load_file(str(ROOT/der["path"]),device="cuda")
                if not torch.equal(jvp(original_d["vector"],layers,caches,co,si),original_d["jvp"]) or not torch.equal(vjp(original_d["vector"],layers,caches,co,si),original_d["vjp"]):raise RuntimeError("derivative changed")
                result["derivative_anchors"]+=1
                n.sync();t=time.perf_counter()
                query=F.normalize(truthless@metric.transform,dim=-1)
                z0=80*(query@metric.table.T);prob=z0.softmax(-1);mean=prob@E
                out,cache,_=forward(mean,layers,history,co,si)
                G,gi=probability_gradient(out,truthless,E,lambda v:vjp(v,layers,cache,co,si))
                n.sync();initialization_seconds=time.perf_counter()-t
                position=torch.tensor([[pos]],device="cuda");rotary=(co[None],si[None])
                def hf(value):
                    hidden=value[None];cache=FrozenPast(history)
                    for layer in prefix.layers:
                        hidden=prefix._hidden(layer(hidden,attention_mask=None,position_ids=position,
                          use_cache=True,cache_position=position[0],position_embeddings=rotary,past_key_values=cache))
                    return hidden[0]
                with torch.enable_grad():
                    variable=prob.detach().requires_grad_();independent=hf(variable@E)
                    objective=1-(F.normalize(independent,dim=-1)*F.normalize(truthless,dim=-1)).sum()
                    expected=torch.autograd.grad(objective,variable)[0].detach()
                reference={"length":length,"position":pos,"gradient_max_absolute_error":float((G-expected).abs().max()),
                  "prefix_max_absolute_error":float((out-independent.detach()).abs().max()),
                  **save(f"gradient_{length}_{pos}",{"initial_logits":z0,"gradient":G,"reference_gradient":expected,
                    "mixture":mean,"output":out,"reference_output":independent.detach(),"target":truthless,"loss":gi["loss"]})}
                result["probability_gradient_references"].append(reference)
                torch.testing.assert_close(G,expected,rtol=5e-4,atol=5e-5)
                torch.testing.assert_close(out,independent.detach(),rtol=5e-4,atol=5e-5)
                reference["passed"]=True;del variable,independent,objective
                result["contexts"].append({"length":length,"position":pos,"source":source,"forward_anchor":old_forward,
                  "derivative_anchor":der,"initialization_and_gradient_seconds":initialization_seconds,
                  "initial_error":float(gi["loss"][0]),"public_history_commit_seconds":public_history_seconds})
                for factor in [2.,1.,.5]:
                  for steps in [8,4,2,1]:
                    reps=[];first=None
                    result["cells"].append({"length":length,"position":pos,"factor":factor,"steps":steps,"repetitions":reps})
                    for rep in range(3):
                        data,stats=trajectory(z0,truthless,E,layers,history,co,si,factor,steps)
                        equal=True if first is None else all(torch.equal(v.cpu(),first[k]) for k,v in data.items())
                        if first is None:first={k:v.cpu() for k,v in data.items()}
                        entry={"rep":rep,**stats,"repeat_equal":equal,"observed_error":float(data["observed_error"][0]),
                          "relative_error":float(data["observed_error"][0]/gi["loss"][0]),
                          "confidence":float(data["confidence"][0]),**save(f"{length}_{pos}_{factor}_{steps}_{rep}",data)}
                        reps.append(entry)
                        if not equal or not stats["passed"]:raise RuntimeError("invalid trajectory; saved")
                        guard()
                if not all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(history,unchanged)):raise RuntimeError("committed past changed")
                print("CAUSAL_VOCAB_CONTEXT_COMPLETE",length,pos,flush=True)
        if len(result["cells"])!=72 or result["forward_anchors"]!=6 or result["derivative_anchors"]!=6 or len(result["probability_gradient_references"])!=6:raise RuntimeError("incomplete matrix")
        for name,sha in result["sources"].items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed during execution")
        result["passed"]=True
    except Exception:
        result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("CAUSAL_VOCAB_PUBLIC_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

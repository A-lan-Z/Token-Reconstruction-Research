"""Public actual-prefix comparison of fixed and backtracked full-vocabulary updates."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage33"))
from probe import n,torch,F,public_parameters,forward,vjp,commit,FrozenPast,trajectory,guard,PrefixWeightMetric,probability_gradient
from descent import solve,STEP_COLUMNS,TRIAL_COLUMNS
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev34_public";DEST=X/"dev34_public_probe.json"
CONFIGS=[("backtrack32","backtrack",32),("fixed32","fixed",32),("backtrack8","backtrack",8),("fixed8","fixed",8)]
@torch.no_grad()
def backtracked(z0,target,E,layers,past,co,si,steps):
    def evaluate(value):
        output,caches,_=forward(value,layers,past,co,si)
        return output,caches
    n.sync();start=time.perf_counter()
    data,states,counts=solve(z0,target,E,evaluate,lambda vector,cache:vjp(vector,layers,cache,co,si),steps)
    n.sync();elapsed=time.perf_counter()-start
    start=time.perf_counter();commit_output,_,current=forward(E.index_select(0,data["token"]),layers,past,co,si)
    committed=commit(past,current);n.sync();commit_seconds=time.perf_counter()-start
    start=time.perf_counter();direct=[];budgets=[]
    for old,new,budget in states:
        a=old.cpu().double().log_softmax(-1);b=new.cpu().double().log_softmax(-1)
        direct.append((a.exp()*(a-b)).sum(-1));budgets.append(budget.cpu().double())
    data["direct_kl"]=torch.cat(direct);data["requested_budget"]=torch.cat(budgets);data["commit_output"]=commit_output
    for i,(key,value) in enumerate(current):data[f"commit_key_{i}"]=key;data[f"commit_value_{i}"]=value
    numeric=all(bool(torch.isfinite(v).all()) for v in data.values())
    monotone=bool((data["step_trace"][:,1]<=data["step_trace"][:,0]).all())
    accepted=data["trial_trace"][:,7].bool()
    armijo=bool((data["trial_trace"][accepted,3]<=data["trial_trace"][accepted,8]).all())
    valid=numeric and monotone and armijo and bool((data["direct_kl"]<=data["requested_budget"]+2e-5).all()) and bool((data["direct_kl"]>=-2e-5).all())
    valid=valid and counts["prefix_forwards"]==len(states)+1 and counts["prefix_vjps"]==steps
    n.sync()
    return data,{**counts,"inference_seconds":elapsed,"emitted_token_commit_seconds":commit_seconds,
      "validation_seconds":time.perf_counter()-start,"numeric_valid":numeric,"monotone":monotone,"armijo":armijo,"passed":valid,
      "commit_prefix_forwards":1,"probability_update_iterations":4,
      "timing_scope":"eager current-token solve with diagnostic logit clones; emitted history commit, FP64 validation and serialization separate"}
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only public study")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism flags")
    old=json.loads((X/"dev33_public_probe.json").read_text())
    cpu=json.loads((X/"dev34_descent_cpu.json").read_text())
    if not old["passed"] or not cpu["passed"]:raise RuntimeError("unqualified prerequisites")
    for d in [old,cpu]:
        for path,sha in d["sources"].items():
            if n.digest(ROOT/path)!=sha:raise RuntimeError("qualified source changed "+path)
    paths=list((ROOT/"scripts/trr0020_stage34").glob("*.py"))+[X/name for name in ["DEV34_DESCENT_PLAN.md","dev34_preflight.json","dev34_descent_cpu.json","dev33_public_probe.json"]]
    result={"task_id":"TRR-0020","kind":"public_full_vocabulary_armijo_descent","sources":old["sources"]|cpu["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "prefix_sha256":old["prefix_sha256"],"config_sha256":old["config_sha256"],"command":sys.argv,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"environment":n.environment(),
      "start_unix":time.time(),"passed":False,"truth_read":False,"scope":"public synthetic fixed past; not reconstruction accuracy",
      "configurations":CONFIGS,"cells":[],"contexts":[],"gradient_references":[],"old_fixed8_anchors":[],
      "step_columns":STEP_COLUMNS,"trial_columns":TRIAL_COLUMNS}
    def save(name,data):
        p=OUT/(name+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard()
        if n.digest(n.ASSETS/"backup/prefix.safetensors")!=old["prefix_sha256"] or n.digest(n.ASSETS/"backup/config.json")!=old["config_sha256"]:raise RuntimeError("asset changed")
        n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);E=prefix.embed_tokens.weight.detach();n.sync()
        result["prefix_preparation_seconds"]=time.perf_counter()-t
        t=time.perf_counter();metric=PrefixWeightMetric(prefix);result["metric_stats"]=metric.build();n.sync()
        result["metric_preparation_seconds"]=time.perf_counter()-t
        for family in ["new","old"]:
          for length in [128,40]:
            if family=="new":
                generator=torch.Generator(device="cpu").manual_seed(34021+length)
                ids=torch.randint(0,len(E),(length,),generator=generator);ids[0]=128000;ids=ids.to("cuda")
                target=prefix.forward_full(ids[None])[0]
                source=save(f"fixture_{family}_{length}",{"public_ids":ids,"public_target":target})
            else:
                source=next(v["source"] for v in old["contexts"] if v["length"]==length)
                if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("old public source changed")
                fixture=n.load_file(str(ROOT/source["path"]),device="cuda");ids=fixture["public_ids"];target=fixture["public_target"]
            if not torch.equal(prefix.forward_full(ids[None])[0],target):raise RuntimeError("public target not repeatable")
            base=prefix.embed_tokens(ids);rotary=prefix.rotary_emb(base[None],torch.arange(length,device="cuda")[None]);cos,sin=rotary[0][0],rotary[1][0]
            selected=[length-1,length//2,1];snapshots={}
            history=[(torch.empty(v["kv_heads"],0,v["head_dim"],device="cuda"),torch.empty(v["kv_heads"],0,v["head_dim"],device="cuda")) for v in layers]
            n.sync();t=time.perf_counter()
            for pos in range(length):
                if pos in selected:snapshots[pos]=history
                _,_,current=forward(base[pos:pos+1],layers,history,cos[pos:pos+1],sin[pos:pos+1]);history=commit(history,current)
            n.sync();history_seconds=time.perf_counter()-t
            for pos in selected:
                past=snapshots[pos];unchanged=[(k.clone(),v.clone()) for k,v in past]
                co,si=cos[pos:pos+1],sin[pos:pos+1];observed=target[pos:pos+1]
                n.sync();t=time.perf_counter();query=F.normalize(observed@metric.transform,dim=-1);z0=80*(query@metric.table.T)
                probability=z0.softmax(-1);mean=probability@E;output,cache,_=forward(mean,layers,past,co,si)
                G,info=probability_gradient(output,observed,E,lambda vector:vjp(vector,layers,cache,co,si));n.sync()
                initialization_seconds=time.perf_counter()-t
                position=torch.tensor([[pos]],device="cuda")
                def hf(value):
                    hidden=value[None];fixed=FrozenPast(past)
                    for layer in prefix.layers:
                        hidden=prefix._hidden(layer(hidden,attention_mask=None,position_ids=position,use_cache=True,
                          cache_position=position[0],position_embeddings=(co[None],si[None]),past_key_values=fixed))
                    return hidden[0]
                with torch.enable_grad():
                    variable=probability.detach().requires_grad_();fresh=hf(variable@E)
                    objective=1-(F.normalize(fresh,dim=-1)*F.normalize(observed,dim=-1)).sum()
                    expected=torch.autograd.grad(objective,variable)[0].detach()
                ref={"family":family,"length":length,"position":pos,
                  "gradient_max_error":float((G-expected).abs().max()),"forward_max_error":float((output-fresh.detach()).abs().max()),
                  **save(f"gradient_{family}_{length}_{pos}",{"logits":z0,"gradient":G,"reference_gradient":expected,"output":output,"reference_output":fresh.detach(),"target":observed})}
                result["gradient_references"].append(ref)
                torch.testing.assert_close(G,expected,rtol=5e-4,atol=5e-5)
                torch.testing.assert_close(output,fresh.detach(),rtol=5e-4,atol=5e-5)
                ref["passed"]=True;del variable,fresh,objective
                result["contexts"].append({"family":family,"length":length,"position":pos,"source":source,
                  "initial_error":float(info["loss"][0]),"initialization_seconds":initialization_seconds,"history_seconds":history_seconds})
                for name,rule,steps in CONFIGS:
                    row={"family":family,"length":length,"position":pos,"method":name,"rule":rule,"steps":steps,"repetitions":[]};result["cells"].append(row)
                    first=None
                    for rep in range(3):
                        if rule=="fixed":data,stats=trajectory(z0,observed,E,layers,past,co,si,2.,steps)
                        else:data,stats=backtracked(z0,observed,E,layers,past,co,si,steps)
                        cpu_data={k:v.cpu() for k,v in data.items()}
                        equal=first is None or (set(first)==set(cpu_data) and all(torch.equal(v,first[k]) for k,v in cpu_data.items()))
                        if first is None:first=cpu_data
                        entry={"rep":rep,**stats,"repeat_equal":equal,"observed_error":float(data["observed_error"][0]),
                          "relative_error":float(data["observed_error"][0]/info["loss"][0]),"confidence":float(data["confidence"][0]),
                          **save(f"{family}_{length}_{pos}_{name}_{rep}",data)}
                        row["repetitions"].append(entry)
                        if not equal or not stats["passed"]:raise RuntimeError("invalid public trajectory; preserved")
                        if family=="old" and name=="fixed8" and rep==0:
                            prior=next(c for c in old["cells"] if c["length"]==length and c["position"]==pos and c["factor"]==2. and c["steps"]==8)["repetitions"][0]
                            if n.digest(ROOT/prior["path"])!=prior["sha256"]:raise RuntimeError("changed old control")
                            saved=n.load_file(str(ROOT/prior["path"]))
                            passed=set(saved)==set(cpu_data) and all(torch.equal(v,saved[k]) for k,v in cpu_data.items())
                            result["old_fixed8_anchors"].append({"length":length,"position":pos,"reference":prior["path"],"passed":passed})
                            if not passed:raise RuntimeError("old fixed8 control changed")
                        guard()
                    if family=="new" and length==128 and pos==127 and name=="backtrack32":
                        result["largest_cell_qualification"]={"passed":True,"peak_reserved":torch.cuda.max_memory_reserved(),"native_context":127,"steps":32}
                        print("LARGEST_DESCENT_CELL_QUALIFIED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
                if not all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(past,unchanged)):raise RuntimeError("history changed")
                print("DESCENT_CONTEXT_COMPLETE",family,length,pos,flush=True)
        if len(result["cells"])!=48 or len(result["gradient_references"])!=12 or len(result["old_fixed8_anchors"])!=6:raise RuntimeError("incomplete matrix")
        for name,sha in result["sources"].items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed during run")
        result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("DESCENT_PUBLIC_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

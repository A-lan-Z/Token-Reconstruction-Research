"""Public common-token comparison of ordinary and norm-preserving mixtures."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource,importlib.util,gc
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage34"))
spec=importlib.util.spec_from_file_location("dev34_qualified_probe",ROOT/"scripts/trr0020_stage34/probe.py")
qualified=importlib.util.module_from_spec(spec);sys.modules[spec.name]=qualified;spec.loader.exec_module(qualified)
n=qualified.n;torch=qualified.torch;F=qualified.F;forward=qualified.forward;vjp=qualified.vjp;commit=qualified.commit
public_parameters=qualified.public_parameters;FrozenPast=qualified.FrozenPast;guard=qualified.guard
probability_gradient=qualified.probability_gradient;PrefixWeightMetric=qualified.PrefixWeightMetric
from transformers import AutoTokenizer
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev35_public";DEST=X/"dev35_public_probe.json"
TEXTS=["a","the","The","Hello","def","import","return","class","0","1","\n"," ","{","}","(","```"]
CONFIGS=[("norm32","normalized",32),("fixed32","fixed",32),("norm8","normalized",8),("fixed8","fixed",8)]
from normalized_mixture import mixture,gradient as norm_gradient
from budget_step import update
@torch.no_grad()
def normalized_trajectory(z0,target,E,norms,layers,past,co,si,steps):
    z=z0.clone();records=[];logits=[z.clone()];budgets=[]
    n.sync();start=time.perf_counter()
    for _ in range(steps):
        p=z.softmax(-1);point,state=mixture(p,E,norms);out,cache,_=forward(point,layers,past,co,si)
        G,info=norm_gradient(out,target,E,norms,state,lambda v:vjp(v,layers,cache,co,si))
        z,step=update(z,G,info["loss"],2.,4)
        records.append(torch.stack([info["loss"][0],p.amax(),step["requested_budget"][0],step["formula_kl"][0],
          step["budget_used"][0],state["raw_squared"][0,0].sqrt(),state["energy"][0,0].sqrt(),state["scale"][0,0]]))
        logits.append(z.clone());budgets.append(step["requested_budget"].clone())
    p=z.softmax(-1);point,state=mixture(p,E,norms);out,_,_=forward(point,layers,past,co,si)
    loss=1-(F.normalize(out,dim=-1)*F.normalize(target,dim=-1)).sum(-1);token=z.argmax(-1)
    n.sync();elapsed=time.perf_counter()-start
    start=time.perf_counter();commit_output,_,current=forward(E.index_select(0,token),layers,past,co,si);committed=commit(past,current)
    n.sync();commit_seconds=time.perf_counter()-start
    start=time.perf_counter();direct=[]
    for old,new in zip(logits[:-1],logits[1:]):
        a=old.cpu().double().log_softmax(-1);b=new.cpu().double().log_softmax(-1)
        direct.append((a.exp()*(a-b)).sum(-1))
    direct=torch.cat(direct);budget=torch.cat(budgets).cpu().double()
    data={"logits":z,"mixture":point,"raw_mixture":state["mean"],"expected_energy":state["energy"],"scale":state["scale"],
      "prefix_output":out,"observed_error":loss,"confidence":p.amax(-1),"token":token,"trace":torch.stack(records),
      "direct_kl":direct,"requested_budget":budget,"commit_output":commit_output}
    for i,(key,value) in enumerate(current):data[f"commit_key_{i}"]=key;data[f"commit_value_{i}"]=value
    numeric=all(bool(torch.isfinite(v).all()) for v in data.values())
    valid=numeric and bool((direct<=budget+2e-5).all()) and bool((direct>=-2e-5).all())
    n.sync()
    return data,{"inference_seconds":elapsed,"emitted_token_commit_seconds":commit_seconds,"validation_seconds":time.perf_counter()-start,
      "numeric_valid":numeric,"passed":valid,"prefix_forwards":steps+1,"prefix_vjps":steps,"vocabulary_mixture_products":steps+1,
      "vocabulary_gradient_products":steps,"commit_prefix_forwards":1,"probability_update_iterations":4,"vocabulary":len(E),"shortlist":None,"model_parameter_updates":0,
      "timing_scope":"eager normalized full-vocabulary updates and final evaluation; logit diagnostic clones included; commit, FP64 validation and final readouts separate"}
@torch.no_grad()
def readouts(data,E,norms,metric):
    n.sync();start=time.perf_counter();point=data["mixture"]
    euclidean=(2*(point@E.T)-norms[None]).argmax(-1)
    query=F.normalize(point@metric.transform,dim=-1);mapped=(query@metric.table.T).argmax(-1)
    n.sync();seconds=time.perf_counter()-start
    return {"euclidean_token":euclidean,"metric_token":mapped},seconds

@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only stress diagnostic")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism flags")
    prior=json.loads((X/"dev34_stress_probe.json").read_text())
    cpu=json.loads((X/"dev35_cpu_reference.json").read_text())
    if not cpu["passed"]:raise RuntimeError("unqualified norm primitive")
    for name,sha in cpu["sources"].items():
        if n.digest(ROOT/name)!=sha:raise RuntimeError("CPU source changed "+name)
    if not prior["passed"]:raise RuntimeError("unqualified update")
    for name,sha in prior["sources"].items():
        if n.digest(ROOT/name)!=sha:raise RuntimeError("qualified source changed "+name)
    paths=list(Path(__file__).parent.glob("*.py"))+[X/name for name in ["DEV35_PLAN.md","dev35_preflight.json","dev35_cpu_reference.json","dev34_stress_probe.json"]]
    token_paths=[p for p in (n.ASSETS/"backup").glob("*") if p.is_file() and ("token" in p.name or p.name=="special_tokens_map.json")]
    result={"task_id":"TRR-0020","scope":"public norm-preserving full-vocabulary interpolation; not benchmark reconstruction",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "environment":n.environment(),"start_unix":time.time(),"passed":False,"truth_read":False,
      "sources":prior["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "prefix_sha256":prior["prefix_sha256"],"config_sha256":prior["config_sha256"],
      "tokenizer_assets":{str(p):n.digest(p) for p in token_paths},"configurations":CONFIGS,
      "public_texts":TEXTS,"contexts":[],"cells":[],"gradient_references":[],"control_anchors":[],"one_hot_checks":[],
      "normalized_trace_columns":["loss","confidence","requested_budget","formula_kl","budget_used","raw_norm","expected_norm","scale"]}
    def save(name,data):
        p=OUT/(name+".safetensors");t=time.perf_counter();n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard()
        if n.digest(n.ASSETS/"backup/prefix.safetensors")!=prior["prefix_sha256"] or n.digest(n.ASSETS/"backup/config.json")!=prior["config_sha256"]:raise RuntimeError("asset changed")
        fixture=prior["fixture"]
        if n.digest(ROOT/fixture["path"])!=fixture["sha256"]:raise RuntimeError("public fixture changed")
        stored=n.load_file(str(ROOT/fixture["path"]));ids=stored["public_ids"].to("cuda");btargets=stored["bf16_target"];ftargets=stored["fp32_target"]
        if prior["public_texts"]!=TEXTS:raise RuntimeError("context set changed")
        result["fixture"]=fixture;result["encoded_public_texts"]=prior["encoded_public_texts"]
        n.sync();t=time.perf_counter();prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);E=prefix.embed_tokens.weight.detach();n.sync();result["prefix_preparation_seconds"]=time.perf_counter()-t
        t=time.perf_counter();norms=E.square().sum(-1);n.sync();result["norm_table_seconds"]=time.perf_counter()-t
        result["norm_table_bytes"]=norms.numel()*norms.element_size()
        t=time.perf_counter();metric=PrefixWeightMetric(prefix);result["metric_stats"]=metric.build();n.sync();result["metric_seconds"]=time.perf_counter()-t
        guard()
        for vertex,row in enumerate(ids):
            one=E.new_zeros((1,len(E)));one[0,row[1]]=1
            point,state=mixture(one,E,norms);expected=E.index_select(0,row[1:2])
            error=float((point-expected).abs().max());torch.testing.assert_close(point,expected,rtol=2e-6,atol=2e-7)
            result["one_hot_checks"].append({"context":vertex,"max_error":error,"passed":True,**save(f"one_hot_{vertex}",{"point":point,"expected":expected,"expected_energy":state["energy"]})})
        positions=torch.arange(2,device="cuda")[None];base=prefix.embed_tokens(ids[0])
        pe=prefix.rotary_emb(base[None],positions);co,si=pe[0][0],pe[1][0]
        empty=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in layers]
        _,_,bos_current=forward(E[128000:128001],layers,empty,co[0:1],si[0:1]);past=commit(empty,bos_current)
        unchanged=[(k.clone(),v.clone()) for k,v in past];position=torch.tensor([[1]],device="cuda")
        for precision,targets in [("fp32",ftargets),("bf16",btargets)]:
          for index,text in enumerate(TEXTS):
            observed=targets[index,1:2].to(device="cuda",dtype=torch.float32)
            n.sync();t=time.perf_counter();query=F.normalize(observed@metric.transform,dim=-1);z0=80*(query@metric.table.T)
            p=z0.softmax(-1);point,state=mixture(p,E,norms);out,cache,_=forward(point,layers,past,co[1:2],si[1:2])
            G,info=norm_gradient(out,observed,E,norms,state,lambda v:vjp(v,layers,cache,co[1:2],si[1:2]));n.sync()
            initialization=time.perf_counter()-t
            def hf(value):
                hidden=value[None];fixed=FrozenPast(past)
                for layer in prefix.layers:
                    hidden=prefix._hidden(layer(hidden,attention_mask=None,position_ids=position,use_cache=True,cache_position=position[0],
                      position_embeddings=(co[None,1:2],si[None,1:2]),past_key_values=fixed))
                return hidden[0]
            with torch.enable_grad():
                variable=p.detach().requires_grad_();mean=variable@E;energy=variable@norms[:,None]
                point_reference=mean*torch.sqrt(energy.clamp_min(1e-24)/mean.square().sum(-1,keepdim=True).clamp_min(1e-24))
                independent=hf(point_reference)
                objective=1-(F.normalize(independent,dim=-1)*F.normalize(observed,dim=-1)).sum()
                expected=torch.autograd.grad(objective,variable)[0].detach()
            ref={"precision":precision,"context":index,"gradient_max_error":float((G-expected).abs().max()),
              "forward_max_error":float((out-independent.detach()).abs().max()),
              **save(f"gradient_{precision}_{index}",{"logits":z0,"gradient":G,"reference_gradient":expected,"output":out,"reference_output":independent.detach(),"target":observed,"mixture":point,"raw_mixture":state["mean"],"expected_energy":state["energy"],"scale":state["scale"]})}
            result["gradient_references"].append(ref)
            torch.testing.assert_close(G,expected,rtol=5e-4,atol=5e-5);torch.testing.assert_close(out,independent.detach(),rtol=5e-4,atol=5e-5)
            ref["passed"]=True;del variable,independent,objective,point_reference,mean,energy
            result["contexts"].append({"precision":precision,"context":index,"public_text":text,"public_token_id":int(ids[index,1]),
              "normal_initial_error":float(info["loss"][0]),"raw_initial_error":next(v["initial_error"] for v in prior["contexts"] if v["precision"]==precision and v["context"]==index),
              "raw_norm":float(state["raw_squared"].sqrt()[0,0]),"expected_norm":float(state["energy"].sqrt()[0,0]),"initial_scale":float(state["scale"][0,0]),"initialization_seconds":initialization,"source":fixture})
            for name,rule,steps in CONFIGS:
                cell={"precision":precision,"context":index,"method":name,"rule":rule,"steps":steps,"repetitions":[]};result["cells"].append(cell)
                first=None
                for rep in range(3):
                    if rule=="fixed":data,stats=qualified.trajectory(z0,observed,E,layers,past,co[1:2],si[1:2],2.,steps)
                    else:data,stats=normalized_trajectory(z0,observed,E,norms,layers,past,co[1:2],si[1:2],steps)
                    if rule=="fixed" and rep==0:
                        reference=next(c for c in prior["cells"] if c["precision"]==precision and c["context"]==index and c["method"]==name)["repetitions"][0]
                        if n.digest(ROOT/reference["path"])!=reference["sha256"]:raise RuntimeError("control changed")
                        saved=n.load_file(str(ROOT/reference["path"]));actual={k:v.cpu() for k,v in data.items()}
                        same=set(saved)==set(actual) and all(torch.equal(v,saved[k]) for k,v in actual.items())
                        result["control_anchors"].append({"precision":precision,"context":index,"method":name,"reference":reference,"passed":same})
                        if not same:raise RuntimeError("unchanged control differs")
                    extra,readout_seconds=readouts(data,E,norms,metric);data.update(extra)
                    stats.update(final_readout_seconds=readout_seconds,final_readout_vocabulary_products=2)
                    current={k:v.cpu() for k,v in data.items()}
                    equal=first is None or (set(current)==set(first) and all(torch.equal(v,first[k]) for k,v in current.items()))
                    if first is None:first=current
                    entry={"rep":rep,**stats,"repeat_equal":equal,"observed_error":float(data["observed_error"][0]),
                      "confidence":float(data["confidence"][0]),"emitted_token_id":int(data["token"][0]),"euclidean_token_id":int(data["euclidean_token"][0]),"metric_token_id":int(data["metric_token"][0]),
                      **save(f"{precision}_{index}_{name}_{rep}",data)}
                    cell["repetitions"].append(entry)
                    if not equal or not stats["passed"]:raise RuntimeError("invalid trajectory; preserved")
                    guard()
            if not all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(past,unchanged)):raise RuntimeError("BOS history changed")
            print("NORMALIZED_CONTEXT_COMPLETE",precision,index,flush=True)
        if len(result["cells"])!=128 or len(result["gradient_references"])!=32 or len(result["control_anchors"])!=64 or len(result["one_hot_checks"])!=16:raise RuntimeError("incomplete matrix")
        result["public_identity_sanity"]=[]
        for precision in ["fp32","bf16"]:
          for name,_,_ in CONFIGS:
            selected=[v for v in result["cells"] if v["precision"]==precision and v["method"]==name]
            for readout,key in [("probability","emitted_token_id"),("euclidean","euclidean_token_id"),("metric","metric_token_id")]:
                result["public_identity_sanity"].append({"precision":precision,"method":name,"readout":readout,"identified":sum(v["repetitions"][0][key]==int(ids[v["context"],1]) for v in selected),"cases":len(selected)})
        for name,sha in result["sources"].items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed during run")
        result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("NORMALIZED_PUBLIC_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

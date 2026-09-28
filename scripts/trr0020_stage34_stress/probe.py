"""Public common-token stress cases for the frozen full-vocabulary descent rule."""
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
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev34_stress";DEST=X/"dev34_stress_probe.json"
TEXTS=["a","the","The","Hello","def","import","return","class","0","1","\n"," ","{","}","(","```"]
CONFIGS=qualified.CONFIGS
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only stress diagnostic")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism flags")
    prior=json.loads((X/"dev34_public_probe.json").read_text())
    if not prior["passed"]:raise RuntimeError("unqualified update")
    for name,sha in prior["sources"].items():
        if n.digest(ROOT/name)!=sha:raise RuntimeError("qualified source changed "+name)
    paths=list(Path(__file__).parent.glob("*.py"))+[X/name for name in ["DEV34_STRESS_PLAN.md","dev34_stress_preflight.json","dev34_public_probe.json"]]
    token_paths=[p for p in (n.ASSETS/"backup").glob("*") if p.is_file() and ("token" in p.name or p.name=="special_tokens_map.json")]
    result={"task_id":"TRR-0020","scope":"public synthetic common-token and arithmetic stress qualification; not benchmark reconstruction",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "environment":n.environment(),"start_unix":time.time(),"passed":False,"truth_read":False,
      "sources":prior["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "prefix_sha256":prior["prefix_sha256"],"config_sha256":prior["config_sha256"],
      "tokenizer_assets":{str(p):n.digest(p) for p in token_paths},"configurations":CONFIGS,
      "public_texts":TEXTS,"contexts":[],"cells":[],"gradient_references":[],"step_columns":qualified.STEP_COLUMNS,"trial_columns":qualified.TRIAL_COLUMNS}
    def save(name,data):
        p=OUT/(name+".safetensors");t=time.perf_counter();n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard()
        if n.digest(n.ASSETS/"backup/prefix.safetensors")!=prior["prefix_sha256"] or n.digest(n.ASSETS/"backup/config.json")!=prior["config_sha256"]:raise RuntimeError("asset changed")
        t=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(n.ASSETS/"backup",local_files_only=True)
        encoded=[tokenizer.encode(text,add_special_tokens=False) for text in TEXTS]
        if any(not ids for ids in encoded):raise RuntimeError("empty public token")
        ids=torch.tensor([[128000,tokens[0]] for tokens in encoded],dtype=torch.long,device="cuda")
        result["tokenizer_seconds"]=time.perf_counter()-t;result["encoded_public_texts"]=encoded
        n.sync();t=time.perf_counter();bf16=n.inherited.load_prefix(torch.bfloat16,asset_root=n.ASSETS/"backup");bf16.config._attn_implementation="eager"
        btargets=torch.stack([bf16.forward_full(row[None])[0].cpu() for row in ids]);n.sync()
        result["bf16_public_target_seconds"]=time.perf_counter()-t
        del bf16;gc.collect();torch.cuda.empty_cache()
        n.sync();t=time.perf_counter();prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);E=prefix.embed_tokens.weight.detach()
        ftargets=torch.stack([prefix.forward_full(row[None])[0].cpu() for row in ids]);n.sync()
        result["fp32_prefix_and_target_seconds"]=time.perf_counter()-t
        fixture=save("public_fixture",{"public_ids":ids,"bf16_target":btargets,"fp32_target":ftargets});result["fixture"]=fixture
        t=time.perf_counter();metric=PrefixWeightMetric(prefix);result["metric_stats"]=metric.build();n.sync();result["metric_seconds"]=time.perf_counter()-t
        positions=torch.arange(2,device="cuda")[None];base=prefix.embed_tokens(ids[0])
        pe=prefix.rotary_emb(base[None],positions);co,si=pe[0][0],pe[1][0]
        empty=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in layers]
        _,_,bos_current=forward(E[128000:128001],layers,empty,co[0:1],si[0:1]);past=commit(empty,bos_current)
        unchanged=[(k.clone(),v.clone()) for k,v in past];position=torch.tensor([[1]],device="cuda")
        for precision,targets in [("fp32",ftargets),("bf16",btargets)]:
          for index,text in enumerate(TEXTS):
            observed=targets[index,1:2].to(device="cuda",dtype=torch.float32)
            n.sync();t=time.perf_counter();query=F.normalize(observed@metric.transform,dim=-1);z0=80*(query@metric.table.T)
            p=z0.softmax(-1);mean=p@E;out,cache,_=forward(mean,layers,past,co[1:2],si[1:2])
            G,info=probability_gradient(out,observed,E,lambda v:vjp(v,layers,cache,co[1:2],si[1:2]));n.sync()
            initialization=time.perf_counter()-t
            def hf(value):
                hidden=value[None];fixed=FrozenPast(past)
                for layer in prefix.layers:
                    hidden=prefix._hidden(layer(hidden,attention_mask=None,position_ids=position,use_cache=True,cache_position=position[0],
                      position_embeddings=(co[None,1:2],si[None,1:2]),past_key_values=fixed))
                return hidden[0]
            with torch.enable_grad():
                variable=p.detach().requires_grad_();independent=hf(variable@E)
                objective=1-(F.normalize(independent,dim=-1)*F.normalize(observed,dim=-1)).sum()
                expected=torch.autograd.grad(objective,variable)[0].detach()
            ref={"precision":precision,"context":index,"gradient_max_error":float((G-expected).abs().max()),
              "forward_max_error":float((out-independent.detach()).abs().max()),
              **save(f"gradient_{precision}_{index}",{"logits":z0,"gradient":G,"reference_gradient":expected,"output":out,"reference_output":independent.detach(),"target":observed})}
            result["gradient_references"].append(ref)
            torch.testing.assert_close(G,expected,rtol=5e-4,atol=5e-5);torch.testing.assert_close(out,independent.detach(),rtol=5e-4,atol=5e-5)
            ref["passed"]=True;del variable,independent,objective
            result["contexts"].append({"precision":precision,"context":index,"public_text":text,"public_token_id":int(ids[index,1]),
              "initial_error":float(info["loss"][0]),"initialization_seconds":initialization,"source":fixture})
            for name,rule,steps in CONFIGS:
                cell={"precision":precision,"context":index,"method":name,"rule":rule,"steps":steps,"repetitions":[]};result["cells"].append(cell)
                first=None
                for rep in range(3):
                    if rule=="fixed":data,stats=qualified.trajectory(z0,observed,E,layers,past,co[1:2],si[1:2],2.,steps)
                    else:data,stats=qualified.backtracked(z0,observed,E,layers,past,co[1:2],si[1:2],steps)
                    current={k:v.cpu() for k,v in data.items()}
                    equal=first is None or (set(current)==set(first) and all(torch.equal(v,first[k]) for k,v in current.items()))
                    if first is None:first=current
                    entry={"rep":rep,**stats,"repeat_equal":equal,"observed_error":float(data["observed_error"][0]),
                      "confidence":float(data["confidence"][0]),"emitted_token_id":int(data["token"][0]),
                      **save(f"{precision}_{index}_{name}_{rep}",data)}
                    cell["repetitions"].append(entry)
                    if not equal or not stats["passed"]:raise RuntimeError("invalid trajectory; preserved")
                    guard()
            if not all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(past,unchanged)):raise RuntimeError("BOS history changed")
            print("COMMON_TOKEN_CONTEXT_COMPLETE",precision,index,flush=True)
        if len(result["cells"])!=128 or len(result["gradient_references"])!=32:raise RuntimeError("incomplete matrix")
        result["public_identity_sanity"]=[]
        for precision in ["fp32","bf16"]:
          for name,_,_ in CONFIGS:
            selected=[v for v in result["cells"] if v["precision"]==precision and v["method"]==name]
            result["public_identity_sanity"].append({"precision":precision,"method":name,"identified":sum(v["repetitions"][0]["emitted_token_id"]==int(ids[v["context"],1]) for v in selected),"cases":len(selected)})
        for name,sha in result["sources"].items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed during run")
        result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("COMMON_TOKEN_PUBLIC_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

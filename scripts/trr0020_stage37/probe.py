"""Public-only direct all-vocabulary response scoring; no hidden labels."""
from pathlib import Path
import sys,os,json,time,subprocess,resource,traceback,copy
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
sys.path.insert(0,str(ROOT/"scripts/trr0017"))
import native as n
from shared_context import SharedContext
from token_reconstruction.public_prefix import PublicPrefixCache
from direct_scores import norms_for_shift,scores
torch=n.torch
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev37_public"
DEST=X/"dev37_public_probe.json"
MODES=["raw","mean","scaled_mean","four_probe"]
LENGTHS=[127,63,15,1]
def guard():
    free,_=torch.cuda.mem_get_info()
    available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
    temp=int(subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],text=True).strip())
    if free<3*2**30 or available<8*2**30 or torch.cuda.max_memory_reserved()>8*2**30 or temp>=80:
        raise RuntimeError("resource margin violated")

@torch.no_grad()
def continuous(prefix,cache,point,independent=False):
    pos=cache.length
    temporary=copy.deepcopy(cache) if independent else PublicPrefixCache(SharedContext(cache,1,len(prefix.layers)),pos)
    hidden=point.reshape(1,1,-1)
    positions=torch.tensor([[pos]],device=hidden.device)
    rotary=prefix.rotary_emb(hidden,positions)
    for layer in prefix.layers:
        hidden=prefix._hidden(layer(hidden,attention_mask=None,position_ids=positions,use_cache=True,
            cache_position=positions[0],position_embeddings=rotary,**{prefix.cache_keyword:temporary.backend}))
    temporary.length=pos+1
    prefix._require_cache_length(temporary,pos+1,"after continuous probe")
    return hidden[0,0].float()

def saved_history(cache):
    return [(q.keys.clone() if q.keys is not None else None,q.values.clone() if q.values is not None else None) for q in cache.backend.layers]
def unchanged(cache,old):
    return all((q.keys is None and k is None and q.values is None and v is None) or (k is not None and v is not None and torch.equal(q.keys,k) and torch.equal(q.values,v)) for q,(k,v) in zip(cache.backend.layers,old))

@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only public diagnostic")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism flags")
    cpu=json.loads((X/"dev37_cpu_reference.json").read_text())
    if not cpu["passed"]:raise RuntimeError("unqualified score arithmetic")
    for path,sha in cpu["sources"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU source changed")
    prior=json.loads((X/"dev34_stress_probe.json").read_text())
    cache_receipt=ROOT/"experiments/TRR-0014/lookup_dev_r1/cache.json"
    cache_meta=json.loads(cache_receipt.read_text())
    table_path=ROOT.parent/"TRR-0014/outputs/TRR-0014/lookup_dev_r1_responses.safetensors"
    dependencies=["scripts/trr0014/native.py","scripts/agent4/common.py","scripts/trr0017/shared_context.py",
      "src/token_reconstruction/public_prefix.py","src/token_reconstruction/a1a2_configuration_search.py",
      "scripts/trr0020_resource_guard.py"]
    paths=[ROOT/v for v in dependencies]+list(Path(__file__).parent.glob("*.py"))+[X/v for v in
      ["DEV37_PLAN.md","dev37_cpu_reference.json","dev37_preflight.json","dev34_stress_probe.json"]]+[cache_receipt]
    result={"task_id":"TRR-0020","scope":"known-public fixed-history numerical diagnostic; not reconstruction accuracy",
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":sys.argv,"environment":n.environment(),"start_unix":time.time(),"passed":False,
      "truth_read":False,"shortlist":None,"model_weight_updates":0,"modes":MODES,"metrics":["cosine","negative_squared_distance"],
      "contexts":[],"cells":[],"cache_anchors":[],"adapter_checks":[],"preparation":{}}
    def save(name,data):
        path=OUT/(name+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard()
        if n.digest(n.ASSETS/"backup/prefix.safetensors")!=cache_meta["prefix_sha256"]:raise RuntimeError("prefix/table mismatch")
        if n.digest(table_path)!=cache_meta["cache_sha256"]:raise RuntimeError("table changed")
        result["prefix_sha256"]=cache_meta["prefix_sha256"]
        result["config_sha256"]=n.digest(n.ASSETS/"backup/config.json")
        result["table_source"]={"path":str(table_path),"sha256":cache_meta["cache_sha256"],"preparation_receipt":str(cache_receipt.relative_to(ROOT)),
          "historical_build_seconds":cache_meta["build_seconds"],"required_rebuild_on_weight_change":True}
        fixture=prior["fixture"]
        if n.digest(ROOT/fixture["path"])!=fixture["sha256"]:raise RuntimeError("public fixture changed")
        ids=n.load_file(str(ROOT/fixture["path"]))["public_ids"][:,1].to("cuda")
        result["fixture_source"]=fixture;result["public_texts"]=prior["public_texts"]
        n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync()
        result["preparation"]["prefix_load_seconds"]=time.perf_counter()-t
        t=time.perf_counter();table=n.load_file(str(table_path),device="cuda")["response"];n.sync()
        result["preparation"]["table_load_seconds"]=time.perf_counter()-t
        V,D=table.shape
        if (V,D)!=(128256,2048):raise RuntimeError("table geometry changed")
        bos=n.new_context(prefix);bos_saved=saved_history(bos)
        for lo in [0,V-256]:
            block=n.candidates(prefix,bos,torch.arange(lo,lo+256,device="cuda"),1)
            equal=torch.equal(block,table[lo:lo+256])
            result["cache_anchors"].append({"first_id":lo,"rows":256,"exact_equal":equal})
            if not equal:raise RuntimeError("native table anchor changed")
        if not unchanged(bos,bos_saved):raise RuntimeError("BOS table validation mutated history")
        t=time.perf_counter()
        e=prefix.embed_tokens.weight.float()
        mean=e.mean(0);second=e.square().mean(0);rms=second.sum().sqrt()
        deviation=(second-mean.square()).clamp_min(0).sqrt()
        generator=torch.Generator().manual_seed(37037)
        gaussian=torch.randn(4,D,generator=generator).to("cuda")
        sample=mean[None]+gaussian*deviation[None]
        sample=sample*(rms/sample.norm(dim=-1,keepdim=True).clamp_min(1e-12))
        points=torch.cat([mean[None],(mean*rms/mean.norm().clamp_min(1e-12))[None],sample]).to(torch.bfloat16)
        del e,gaussian,sample
        norms=table.square().sum(-1)
        reference=torch.stack([continuous(prefix,bos,point) for point in points])
        n.sync();result["preparation"]["statistics_and_reference_probes_seconds"]=time.perf_counter()-t
        result["preparation"]["reference_probe_prefix_forwards"]=6
        result["probe_assets"]=save("probe_assets",{"points":points,"reference_responses":reference,"mean_embedding":mean,
            "coordinate_standard_deviation":deviation,"rms_embedding_norm":rms.reshape(1)})
        for pastlen in LENGTHS:
            generator=torch.Generator().manual_seed(37037+pastlen)
            history=torch.cat([torch.tensor([128000]),torch.randint(1000,120000,(pastlen-1,),generator=generator)]).to("cuda")
            cache=prefix.new_cache()
            n.sync();t=time.perf_counter();prefix.run_cached(history[None],cache,0);n.sync()
            history_seconds=time.perf_counter()-t;original=saved_history(cache)
            # Native targets are public observations; true IDs never enter a score function.
            targets=[]
            for token in ids:
                native=n.candidates(prefix,cache,token.reshape(1),pastlen)[0]
                adapter=continuous(prefix,cache,prefix.embed_tokens.weight[token])
                passed=torch.equal(native,adapter)
                result["adapter_checks"].append({"past":pastlen,"kind":"actual_token","exact_equal":passed})
                if not passed:raise RuntimeError("native token adapter differs")
                targets.append(native)
            target=torch.stack(targets)
            for point in points:
                current=continuous(prefix,cache,point)
                independent=continuous(prefix,cache,point,True)
                passed=torch.equal(current,independent)
                result["adapter_checks"].append({"past":pastlen,"kind":"continuous_point","exact_equal":passed})
                if not passed:raise RuntimeError("continuous reference differs")
            context={"past":pastlen,"history_seed":37037+pastlen,"history_seconds":history_seconds,
              **save("fixture_"+str(pastlen),{"history":history,"public_ids":ids,"targets":target})}
            if pastlen==1:
                context["batch256_vs_batch1_max_error"]=float((table.index_select(0,ids)-target).abs().max())
                context["batch256_vs_batch1_mean_cosine_error"]=float((1-torch.nn.functional.cosine_similarity(table.index_select(0,ids),target)).mean())
            result["contexts"].append(context)
            for mode in MODES:
                row={"past":pastlen,"mode":mode,"repetitions":[]};result["cells"].append(row)
                first=None
                for rep in range(3):
                    n.sync();t=time.perf_counter()
                    if mode=="raw":shift=torch.zeros(D,device="cuda");calls=0
                    elif mode=="mean":shift=continuous(prefix,cache,points[0])-reference[0];calls=1
                    elif mode=="scaled_mean":shift=continuous(prefix,cache,points[1])-reference[1];calls=1
                    else:shift=torch.stack([continuous(prefix,cache,points[j])-reference[j] for j in range(2,6)]).mean(0);calls=4
                    n.sync();correction_seconds=time.perf_counter()-t
                    n.sync();t=time.perf_counter();square,unstable=norms_for_shift(table,norms,shift);n.sync()
                    norm_seconds=time.perf_counter()-t
                    cosine_rows=[];distance_rows=[];emitted=[];timing=[];transfer_seconds=0.
                    for observation in target:
                        n.sync();t=time.perf_counter()
                        cosine,distance=scores(table,square,shift,observation,unstable)
                        token=torch.stack([cosine.argmax(),distance.argmax()])
                        n.sync();timing.append(time.perf_counter()-t)
                        t=time.perf_counter()
                        cosine_rows.append(cosine.cpu());distance_rows.append(distance.cpu());emitted.append(token.cpu())
                        transfer_seconds+=time.perf_counter()-t
                    arrays={"cosine":torch.stack(cosine_rows),"negative_squared_distance":torch.stack(distance_rows),
                      "tokens":torch.stack(emitted),"shift":shift.cpu(),"corrected_squared_norms":square.cpu()}
                    if not all(bool(torch.isfinite(v).all()) for v in arrays.values()):raise RuntimeError("nonfinite direct scores")
                    equal=first is None or all(torch.equal(v,first[k]) for k,v in arrays.items())
                    if first is None:first=arrays
                    entry={"rep":rep,"exact_repeat":equal,"correction_prefix_forwards":calls,
                      "correction_seconds":correction_seconds,"norm_adjustment_seconds":norm_seconds,
                      "per_observation_scoring_seconds":timing,"diagnostic_score_transfer_seconds":transfer_seconds,
                      "numerical_norm_fallback_rows":len(unstable),
                      "cost_scope":"Each future token needs correction+norm adjustment+one observation score. History construction, emitted commit and diagnostic full-score transfers are separate."}
                    if rep==0:entry.update(save(str(pastlen)+"_"+mode,arrays))
                    row["repetitions"].append(entry)
                    if not equal:raise RuntimeError("nonrepeatable full scores")
                    if not unchanged(cache,original):raise RuntimeError("committed history changed")
                    guard()
                print("DIRECT_ALL_VOCABULARY_COMPLETE",pastlen,mode,flush=True)
            if pastlen==127:
                result["largest_cell_qualification"]={"passed":True,"past":127,"vocabulary":V,"all_modes_repeated":True,
                  "peak_reserved":torch.cuda.max_memory_reserved()}
                print("LARGEST_DIRECT_CELL_QUALIFIED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        if len(result["cells"])!=16 or len(result["adapter_checks"])!=88:raise RuntimeError("incomplete diagnostic")
        for path,sha in result["sources"].items():
            if n.digest(ROOT/path)!=sha:raise RuntimeError("source changed during run")
        result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("DIRECT_PUBLIC_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

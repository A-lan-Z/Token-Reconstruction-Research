"""Public synthetic one-step comparison; no reconstruction truth is accessed."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage31"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage29"))
from support import n,torch,MirrorOptimizer,gradient_reference,guard,TRACE_KEYS,binding as old_binding
from discrete_soft import DiscreteSoftVocabulary
from fast_soft import BF16Mixture
from mirror_step import update as original
from kl_step import update as original_calibrated
from budget_step import update as budget_update,path_update
from torch.nn import functional as F
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev32_public";DEST=X/"dev32_public_probe.json"

def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only public probe")
    OUT.mkdir(parents=True)
    cpu=json.loads((X/"dev32_cpu_reference.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU qualification failed")
    for path,sha in cpu["sources"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU-qualified source changed")
    paths=list((ROOT/"scripts/trr0020_stage32").glob("*.py"))+[ROOT/name for name in cpu["sources"]]
    paths += [X/name for name in ["DEV32_PLAN.md","dev32_preflight.json","dev32_cpu_reference.json","dev31_public_probe_r1.json"]]
    sources={**old_binding()["sources"],**{str(p.relative_to(ROOT)):n.digest(p) for p in paths}}
    result={"task_id":"TRR-0020","kind":"public_one_step_observed_error_KL_budgets","sources":sources,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":sys.argv,"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "baseline_artifact_anchors":0,"scalar_controls":[],"config_sha256":n.digest(n.ASSETS/"backup/config.json"),"start_unix":time.time(),"passed":False,
      "truth_read":False,"shortlist":None,"model_weight_updates":0,"contexts":[],"cells":[],
      "work_per_context":{"warm_prefix_forwards":"warm+1","warm_prefix_backwards":"warm",
        "diagnostic_base_gradient_forwards":1,"diagnostic_base_gradient_backwards":1,
        "independent_gradient_reference_forwards":2,"independent_gradient_reference_backwards":2,
        "scalar_control_forwards":0,"scalar_control_backwards":0},
      "work_per_update":{"prefix_forwards":0,"prefix_backwards":0,"validation_prefix_forwards":1,
        "calibrated_kl_path_evaluations":"iterations+1","calibrated_path_full_vocabulary_reductions_per_evaluation":3},
      "timing_scope":"eager update-only, validation and IO separate; not end-to-end decoder runtime"}
    def save(label,data):
        path=OUT/(label+".safetensors");start=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-start}
    try:
        guard();n.sync();start=time.perf_counter();prefix=n.load_prefix();n.sync()
        result["prefix_preparation_seconds"]=time.perf_counter()-start
        start=time.perf_counter();engine=MirrorOptimizer(prefix,64,.01);n.sync()
        result["engine_and_lookup_preparation_seconds"]=time.perf_counter()-start
        fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda")
        fixture[0,0]=128000
        for length in [128,40]:
          with torch.no_grad():h=prefix.forward_full(fixture[:,:length])[0].float()
          for warm in [64,0]:
            engine.steps=warm;engine.stage="warm";warm_output,warm_stats=DiscreteSoftVocabulary.decode(engine,h)
            with torch.no_grad():
                z=engine.logits[:length-1].clone()
                p=torch.softmax(z,-1);mean=BF16Mixture.apply(p,engine.weight)
                pe,mask=engine.geometry[length]
                baseline=engine.forward(torch.cat([engine.bos,mean[None]],1),pe,mask)
                initial_error=(1-F.cosine_similarity(baseline[:,1:],h[None,1:],dim=-1))[0]
            anchors=list(X.glob(f"dev29_qualification_warm{warm}_tau0.01_*.json"))
            if len(anchors)!=1:raise RuntimeError("ambiguous warm anchor")
            anchor_record=json.loads(anchors[0].read_text())
            anchor=next(v for v in anchor_record["runs"] if v["length"]==length and v["repetition"]==0)
            if n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise RuntimeError("anchor changed")
            archived=n.load_file(str(ROOT/anchor["path"]))
            warm_equal=(all(torch.equal(v,archived["warm_"+k]) for k,v in warm_output.items())
              and all(warm_stats[k]==anchor["stats"]["warm"][k] for k in TRACE_KEYS)
              and torch.equal(mean.cpu(),archived["initial_embedding"]))
            if not warm_equal:raise RuntimeError("original warm state changed")
            engine.stage="mirror";engine.reset_mirror_statistics();engine.gradient.zero_()
            n.sync();start=time.perf_counter();loss=engine.evaluate(length);loss.backward();n.sync()
            gradient_seconds=time.perf_counter()-start
            del loss  # Release default-stream AccumulateGrad before capturing another length.
            g=engine.gradient[:length-1].detach().clone()
            check=gradient_reference(engine,h)
            if not check["passed"] or not torch.equal(g,engine.gradient[:length-1]):raise RuntimeError("probability gradient changed")
            zcpu=z.cpu().double();logp_cpu=zcpu.log_softmax(-1);prob_cpu=logp_cpu.exp();del zcpu
            previous=json.loads((X/"dev31_public_probe_r1.json").read_text())
            context={"length":length,"warm":warm,"warm_equal":warm_equal,"anchor":anchor["path"],
              "anchor_sha256":anchor["sha256"],"warm_stats":warm_stats,"gradient_reference":check,
              "gradient_seconds":gradient_seconds,"initial_observed_error":float(initial_error.mean()),
              "baseline":save(f"baseline_{length}_{warm}",{"public_ids":fixture[0,:length],"target":h,
                "mixture":mean,"prefix_output":baseline,"observed_error":initial_error,
                "gradient_samples":g[torch.tensor([0,(length-1)//2,length-2],device="cuda")][:,torch.tensor([0,7,64128,128255],device="cuda")]})}
            result["contexts"].append(context)
            previous_context=next(v for v in previous["contexts"] if v["length"]==length and v["warm"]==warm)
            if context["baseline"]["sha256"]!=previous_context["baseline"]["sha256"]:raise RuntimeError("baseline changed")
            result["baseline_artifact_anchors"]+=1
            original_logits,oi=original_calibrated(z,g,.01)
            same_logits,si=path_update(z,g,.01,16)
            same=torch.equal(original_logits,same_logits) and all(torch.equal(v,si[k]) for k,v in oi.items())
            result["scalar_controls"].append({"length":length,"warm":warm,"full_logits_and_diagnostics_equal":same})
            if not same:raise RuntimeError("scalar16control changed")
            del original_logits,same_logits,oi,si
            for factor in [.5,1.,2.]:
              for iterations in [16,8,4]:
                first=None;first_data=None;reps=[]
                result["cells"].append({"length":length,"warm":warm,"factor":factor,"iterations":iterations,"repetitions":reps})
                for rep in range(3):
                    n.sync();start=time.perf_counter();new,info=budget_update(z,g,initial_error,factor,iterations);n.sync()
                    seconds=time.perf_counter()-start
                    repeat=True if first is None else torch.equal(first,new)
                    if first is None:first=new.clone()
                    start=time.perf_counter()
                    with torch.no_grad():
                        probability=torch.softmax(new,-1)
                        mixture=BF16Mixture.apply(probability,engine.weight)
                        output=engine.forward(torch.cat([engine.bos,mixture[None]],1),pe,mask)
                        error=(1-F.cosine_similarity(output[:,1:],h[None,1:],dim=-1))[0]
                        confidence=probability.amax(-1)
                        direct_kl=(prob_cpu*(logp_cpu-new.cpu().double().log_softmax(-1))).sum(-1)
                        budget=info["requested_budget"].cpu().double()
                        usage=direct_kl[budget>0]/budget[budget>0]
                        data={"mixture":mixture,"prefix_output":output,"observed_error":error,
                          "confidence":confidence,"tokens":new.argmax(-1),"direct_kl":direct_kl,
                          **{"solve_"+k:v for k,v in info.items()}}
                    n.sync();validation_seconds=time.perf_counter()-start
                    numeric=bool(torch.isfinite(new).all()) and all(bool(torch.isfinite(v).all()) for v in data.values())
                    data_equal=True if first_data is None else all(torch.equal(v.cpu(),first_data[k]) for k,v in data.items())
                    if first_data is None:first_data={k:v.cpu() for k,v in data.items()}
                    artifact=save(f"{length}_{warm}_{factor}_{iterations}_{rep}",data)
                    passed=repeat and data_equal and numeric and bool((direct_kl<=budget+2e-5).all())
                    passed=passed and bool((info["initial_upper_kl"][info["active"]]>=info["target"][info["active"]]-2e-5).all())
                    row={"rep":rep,"update_seconds":seconds,"validation_seconds":validation_seconds,
                      "full_logits_repeat_equal":repeat,"saved_data_repeat_equal":data_equal,
                      "finite":numeric,"passed":passed,"observed_error":float(error.mean()),
                      "positions_error_improved":int((error<initial_error).sum()),"positions":length-1,
                      "relative_observed_error":float(error.mean()/initial_error.mean()),
                      "mean_confidence":float(confidence.mean()),"kl_min":float(direct_kl.min()),
                      "kl_mean":float(direct_kl.mean()),"kl_max":float(direct_kl.max()),
                      "budget_min":float(budget.min()),"budget_mean":float(budget.mean()),"budget_max":float(budget.max()),
                      "nonzero_budget_positions":int((budget>0).sum()),
                      "usage_min":float(usage.min()) if len(usage) else None,
                      "usage_mean":float(usage.mean()) if len(usage) else None,
                      "usage_max":float(usage.max()) if len(usage) else None,**artifact}
                    reps.append(row)
                    if not passed:raise RuntimeError("invalid public result; preserved")
                    guard()
                del first,first_data,new,probability
            del z,g,logp_cpu,prob_cpu
            print("BUDGET_CONTEXT_COMPLETE",length,warm,flush=True)
        if len(result["cells"])!=36 or len(result["contexts"])!=4 or result["baseline_artifact_anchors"]!=4:raise RuntimeError("incomplete public matrix")
        for name,sha in sources.items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed during execution")
        result["passed"]=True
    except Exception:
        result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),
          peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("BUDGET_PUBLIC_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

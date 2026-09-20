"""Public synthetic one-step comparison; no reconstruction truth is accessed."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage29"))
from support import n,torch,MirrorOptimizer,gradient_reference,guard,TRACE_KEYS,binding as old_binding
from discrete_soft import DiscreteSoftVocabulary
from fast_soft import BF16Mixture
from mirror_step import update as original
from kl_step import update as calibrated,ITERATIONS
from torch.nn import functional as F
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev31_public_r1";DEST=X/"dev31_public_probe_r1.json"

def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only public probe")
    OUT.mkdir(parents=True)
    cpu=json.loads((X/"dev31_cpu_reference.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU qualification failed")
    for path,sha in cpu["sources"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU-qualified source changed")
    paths=list((ROOT/"scripts/trr0020_stage31").glob("*.py"))
    paths += [X/name for name in ["DEV31_PROSPECTIVE.md","DEV31_EXECUTION.md","dev31_preflight_r1.json","dev31_cpu_reference.json","DEV31_REPAIR.md","dev31_public_probe.json","dev31_guard.json"]]
    sources={**old_binding()["sources"],**{str(p.relative_to(ROOT)):n.digest(p) for p in paths}}
    result={"task_id":"TRR-0020","kind":"public_one_step_forward_KL_calibration","sources":sources,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":sys.argv,"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "completed_attempt_anchor_checks":0,"config_sha256":n.digest(n.ASSETS/"backup/config.json"),"start_unix":time.time(),"passed":False,
      "truth_read":False,"shortlist":None,"model_weight_updates":0,"contexts":[],"cells":[],
      "work_per_context":{"warm_prefix_forwards":"warm+1","warm_prefix_backwards":"warm",
        "diagnostic_base_gradient_forwards":1,"diagnostic_base_gradient_backwards":1,
        "independent_gradient_reference_forwards":2,"independent_gradient_reference_backwards":2,
        "old_control_forwards":3,"old_control_backwards":3},
      "work_per_update":{"prefix_forwards":0,"prefix_backwards":0,"validation_prefix_forwards":1,
        "calibrated_kl_path_evaluations":ITERATIONS+1,"calibrated_path_full_vocabulary_reductions_per_evaluation":3},
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
            old_attempt=json.loads((X/"dev31_public_probe.json").read_text())
            context={"length":length,"warm":warm,"warm_equal":warm_equal,"anchor":anchor["path"],
              "anchor_sha256":anchor["sha256"],"warm_stats":warm_stats,"gradient_reference":check,
              "gradient_seconds":gradient_seconds,"initial_observed_error":float(initial_error.mean()),
              "baseline":save(f"baseline_{length}_{warm}",{"public_ids":fixture[0,:length],"target":h,
                "mixture":mean,"prefix_output":baseline,"observed_error":initial_error,
                "gradient_samples":g[torch.tensor([0,(length-1)//2,length-2],device="cuda")][:,torch.tensor([0,7,64128,128255],device="cuda")]})}
            result["contexts"].append(context)
            old_context=next((v for v in old_attempt["contexts"] if v["length"]==length and v["warm"]==warm),None)
            if old_context is not None:
                if context["baseline"]["sha256"]!=old_context["baseline"]["sha256"]:raise RuntimeError("repair changed baseline")
                result["completed_attempt_anchor_checks"]+=1
            for tau in [.01,.1,1.]:
                engine.tau=tau;engine.stage="mirror"
                with torch.no_grad():engine.logits[:length-1].copy_(z);engine.reset_mirror_statistics()
                engine.train_step(length);control=engine.logits[:length-1].detach().clone()
                if not torch.equal(engine.gradient[:length-1],g):raise RuntimeError("old gradient control changed")
                for name,fn in [("span4",original),("forward_kl",calibrated)]:
                    first=None;first_data=None;reps=[]
                    result["cells"].append({"length":length,"warm":warm,"tau":tau,"rule":name,"repetitions":reps})
                    for rep in range(3):
                        n.sync();start=time.perf_counter();new,info=fn(z,g,tau);n.sync()
                        seconds=time.perf_counter()-start
                        repeat=True if first is None else torch.equal(first,new)
                        if first is None:first=new.clone()
                        control_equal=True if name!="span4" else torch.equal(new,control)
                        start=time.perf_counter()
                        with torch.no_grad():
                            probability=torch.softmax(new,-1)
                            mixture=BF16Mixture.apply(probability,engine.weight)
                            output=engine.forward(torch.cat([engine.bos,mixture[None]],1),pe,mask)
                            error=(1-F.cosine_similarity(output[:,1:],h[None,1:],dim=-1))[0]
                            confidence=probability.amax(-1)
                            direct_kl=(prob_cpu*(logp_cpu-new.cpu().double().log_softmax(-1))).sum(-1)
                            data={"mixture":mixture,"prefix_output":output,"observed_error":error,
                              "confidence":confidence,"tokens":new.argmax(-1),"direct_kl":direct_kl,
                              **{"solve_"+k:v for k,v in info.items()}}
                        n.sync();validation_seconds=time.perf_counter()-start
                        numeric=bool(torch.isfinite(new).all()) and all(bool(torch.isfinite(v).all()) for v in data.values())
                        data_equal=True if first_data is None else all(torch.equal(v.cpu(),first_data[k]) for k,v in data.items())
                        if first_data is None:first_data={k:v.cpu() for k,v in data.items()}
                        artifact=save(f"{length}_{warm}_{tau}_{name}_{rep}",data)
                        passed=repeat and data_equal and control_equal and numeric
                        if name=="forward_kl":
                            passed=passed and bool((direct_kl<=tau+2e-5).all()) and bool((info["initial_upper_kl"][info["active"]]>=tau*(1-1e-4)-2e-5).all())
                        row={"rep":rep,"update_seconds":seconds,"validation_seconds":validation_seconds,
                          "full_logits_repeat_equal":repeat,"saved_data_repeat_equal":data_equal,"old_control_equal":control_equal,
                          "finite":numeric,"passed":passed,"observed_error":float(error.mean()),
                          "positions_error_improved":int((error<initial_error).sum()),"positions":length-1,
                          "relative_observed_error":float(error.mean()/initial_error.mean()),
                          "mean_confidence":float(confidence.mean()),"kl_min":float(direct_kl.min()),
                          "kl_mean":float(direct_kl.mean()),"kl_max":float(direct_kl.max()),**artifact}
                        reps.append(row)
                        old_cell=next((v for v in old_attempt["cells"] if v["length"]==length and v["warm"]==warm and v["tau"]==tau and v["rule"]==name),None)
                        if old_cell is not None:
                            old_row=old_cell["repetitions"][rep]
                            if row["sha256"]!=old_row["sha256"]:raise RuntimeError("repair changed saved result")
                            result["completed_attempt_anchor_checks"]+=1
                        if not passed:raise RuntimeError("invalid public result; preserved")
                        guard()
                    del first,first_data,new,probability
                del control
            del z,g,logp_cpu,prob_cpu
            print("KL_CONTEXT_COMPLETE",length,warm,flush=True)
        if len(result["cells"])!=24 or len(result["contexts"])!=4 or result["completed_attempt_anchor_checks"]!=38:raise RuntimeError("incomplete public matrix")
        for name,sha in sources.items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed during execution")
        result["passed"]=True
    except Exception:
        result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),
          peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("KL_PUBLIC_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

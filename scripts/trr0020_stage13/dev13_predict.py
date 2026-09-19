"""Prospective full-vocabulary constrained embedding experiment; no evaluation labels imported."""
from pathlib import Path
import sys,json,time,gc,subprocess,resource,os
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5","scripts/trr0020_stage6r1"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
from constrained_embedding import ConstrainedEmbedding,CONFIGS,reference_tests
torch=n.torch
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
    raise RuntimeError("required deterministic environment missing")
torch.use_deterministic_algorithms(True)
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev13"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"

def binding():
    paths=list((ROOT/"scripts/trr0020_stage13").glob("*.py"))+[X/"DEV13_PLAN.md"]+[ROOT/p for p in [
      "scripts/trr0020_stage6r1/fast_soft.py","scripts/trr0020_stage5/graph_soft.py",
      "scripts/trr0020/soft_vocabulary.py","scripts/trr0020_stage3/reset_soft.py",
      "scripts/trr0014/native.py","scripts/agent4/common.py","scripts/trr0020_resource_guard.py",
      "src/token_reconstruction/public_prefix.py","src/token_reconstruction/prefix_weight_metric.py"]]
    return {"execution_flags":{"deterministic_algorithms":True,"cublas_workspace_config":os.environ["CUBLAS_WORKSPACE_CONFIG"],"allow_tf32":False},
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in sorted(paths)},
      "observations_sha256":n.digest(INPUT/"observations.safetensors"),
      "metadata_sha256":n.digest(INPUT/"metadata.json"),"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors")}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bound=binding();env=n.environment();start=time.time()
    n.write(X/f"dev13_math_test_{time.time_ns()}.json",{"binding":bound,**reference_tests(),"passed":True})
    n.guard();load_start=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-load_start
    data=n.load_file(str(INPUT/"observations.safetensors"))
    selected=[];counts={}
    for row in json.loads((INPUT/"metadata.json").read_text()):
        key=(row["condition"],row["group"]);counts.setdefault(key,0)
        if counts[key]<2:
            selected.append(row);counts[key]+=1
    assert len(selected)==8 and len(CONFIGS)==12
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200033)).to("cuda")
    fixture[0,0]=128000
    with torch.no_grad():h=prefix.forward_full(fixture)[0].float()
    entries=[];setups={}
    for name,strength,lr in CONFIGS:
        n.guard();n.sync();t=time.perf_counter();engine=ConstrainedEmbedding(prefix,strength,lr)
        n.sync();setups[name]={"engine_seconds":time.perf_counter()-t}
        runs=[];predictions=[]
        for repetition in range(3):
            current,stats=engine.decode(h);n.guard()
            raw=X/f"dev13_qualification_{name}_{repetition}_{time.time_ns()}.safetensors"
            n.save_file(current,str(raw));predictions.append(current)
            runs.append({"path":str(raw.relative_to(ROOT)),"sha256":n.digest(raw),"stats":stats})
        comparisons=[]
        for repetition in [1,2]:
            one=runs[0]["stats"]["loss_trace"];two=runs[repetition]["stats"]["loss_trace"]
            comparisons.append({"repetition":repetition,
              "tokens_equal":all(torch.equal(predictions[0][k],predictions[repetition][k]) for k in predictions[0]),
              "losses_equal":all(runs[0]["stats"][key]==runs[repetition]["stats"][key] for key in ["loss_trace","observed_error_trace","nearest_distance_trace"]),"maximum_loss_difference":max(abs(a-b) for a,b in zip(one,two)),
              "different_loss_entries":sum(a!=b for a,b in zip(one,two))})
        n.write(X/f"dev13_repeatability_{name}_{time.time_ns()}.json",
          {"binding":bound,"runs":runs,"comparisons":comparisons,"environment":env,"setup":setups[name]})
        if not all(c["tokens_equal"] and c["losses_equal"] for c in comparisons):
            raise RuntimeError("nonrepeatable largest geometry; all traces preserved")
        free,total=torch.cuda.mem_get_info()
        if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:
            raise RuntimeError("insufficient qualified memory margin")
        qualification={"binding":bound,"environment":env,"repeated_outputs_and_loss_trace_equal":True,
          "repeatability":comparisons,"stats":runs[0]["stats"],"free_bytes":free,
          "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
          "synthetic_correct":{k:int((v[1:]==fixture[0,1:].cpu()).sum()) for k,v in predictions[0].items()}}
        n.write(X/f"dev13_qualification_{name}_{time.time_ns()}.json",qualification)
        print("QUALIFIED",name,round(runs[-1]["stats"]["total_seconds"],4),qualification["synthetic_correct"],flush=True)
        for row in selected:
            path=OUT/(row["id"]+"__"+name+".safetensors");rp=path.with_suffix(".json")
            if rp.exists():
                e=json.loads(rp.read_text())
                if e["binding"]!=bound or n.digest(path)!=e["sha256"]:raise RuntimeError("changed resume")
                entries.append(e);continue
            if path.exists():raise RuntimeError("orphan output")
            output,stats=engine.decode(data[row["id"]]);n.guard()
            n.save_file(output,str(path))
            e={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"binding":bound,
               "stats":stats,"frozen_unix":time.time()}
            n.write(rp,e);entries.append(e)
            print("FROZEN",row["id"],name,round(stats["total_seconds"],4),flush=True)
        setups[name]["capture_events"]=engine.capture_events
        del engine;gc.collect();torch.cuda.empty_cache()
    if binding()!=bound:raise RuntimeError("source changed")
    if len(entries)!=96:raise RuntimeError("incomplete matrix")
    n.write(X/"dev13_freeze.json",{"task_id":"TRR-0020","binding":bound,"environment":env,"entries":entries,
      "selected_records":selected,"configurations":CONFIGS,"setups":setups,"truth_read":False,
      "prefix_load_seconds":prefix_seconds,"start_unix":start,"end_unix":time.time(),
      "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
      "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"command":sys.argv})
    print("COMPLETE96CELLFREEZE",flush=True)
if __name__=="__main__":main()

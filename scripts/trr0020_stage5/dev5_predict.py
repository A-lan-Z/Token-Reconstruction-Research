from pathlib import Path
import sys,json,time,gc,subprocess,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"));import native as n
sys.path.insert(0,str(ROOT/"scripts/trr0020"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage3"))
from graph_soft import GraphSoftVocabulary
from reset_soft import ResetSoftVocabulary
torch=n.torch;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev5"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
CONFIGS=[("cosine",.6,False)]+[("cosine",r,True) for r in [.3,.6,1.]]+[(o,r,True) for o in ["mse","mixed","white"] for r in [.3,.6]]
def binding():
    paths=list((ROOT/"scripts/trr0020_stage5").glob("*.py"))+[X/"DEV5_PLAN.md",
      ROOT/"scripts/trr0020/soft_vocabulary.py",ROOT/"scripts/trr0020_stage3/reset_soft.py",
      ROOT/"scripts/trr0014/native.py",ROOT/"scripts/agent4/common.py",
      ROOT/"src/token_reconstruction/public_prefix.py",ROOT/"src/token_reconstruction/prefix_weight_metric.py"]
    return {"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "observations_sha256":n.digest(INPUT/"observations.safetensors"),
      "metadata_sha256":n.digest(INPUT/"metadata.json"),"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors")}
def main():
    OUT.mkdir(parents=True,exist_ok=True);bound=binding();env=n.environment();start=time.time()
    prefix=n.load_prefix();data=n.load_file(str(INPUT/"observations.safetensors"))
    selected=[];counts={}
    for row in json.loads((INPUT/"metadata.json").read_text()):
        key=(row["condition"],row["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(row);counts[key]+=1
    gen=torch.Generator().manual_seed(200025)
    fixture=torch.randint(256,128000,(1,128),generator=gen).to("cuda");fixture[0,0]=128000
    with torch.no_grad():h=prefix.forward_full(fixture)[0].float()
    # Native reference for new synthetic sequence and each actual length.
    reference=ResetSoftVocabulary(prefix);reference_out={}
    for length in [128,40,83]:
        reference_out[length]=reference.decode(h[:length],80.,.6,.98)
    del reference;gc.collect();torch.cuda.empty_cache()
    entries=[];setups={}
    for objective,lr,tf32 in CONFIGS:
        name=f"graph_{objective}_{lr}_{'tf32' if tf32 else 'fp32'}"
        n.sync();t=time.perf_counter();engine=GraphSoftVocabulary(prefix,objective,lr,tf32)
        n.sync();setups[name]={"engine_seconds":time.perf_counter()-t}
        qualification=[]
        lengths=[128,40,83,128] if not tf32 else [128]
        for length in lengths:
            output,stats=engine.decode(h[:length]);n.guard()
            row={"length":length,"stats":stats,"synthetic_correct":{k:int((v[1:]==fixture[0,1:length].cpu()).sum()) for k,v in output.items()}}
            if not tf32:
                original,original_stats=reference_out[length]
                row["reference_tokens_equal"]={k:bool(torch.equal(output[k],original[k])) for k in output}
                row["maximum_loss_difference"]=max(abs(a-b["objective"]) for a,b in zip(stats["loss_trace"],original_stats["trace"]))
                if not all(row["reference_tokens_equal"].values()):
                    n.write(X/f"dev5_failed_equivalence_{time.time_ns()}.json",{"binding":bound,"row":row})
                    raise RuntimeError("graph is not an equivalent execution; preserve and investigate")
            qualification.append(row);print("qualified",name,length,round(stats["total_seconds"],3),row.get("maximum_loss_difference"),flush=True)
        free,total=torch.cuda.mem_get_info()
        if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError("insufficient qualified margin")
        n.write(X/f"dev5_qualification_{name}_{time.time_ns()}.json",{"binding":bound,"rows":qualification,"setup":setups[name],
          "environment":env,"free_bytes":free,"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
        for row in selected:
            path=OUT/(row["id"]+"__"+name+".safetensors");rp=path.with_suffix(".json")
            if rp.exists():
                e=json.loads(rp.read_text())
                if e["binding"]!=bound or n.digest(path)!=e["sha256"]:raise RuntimeError("changed resume")
                entries.append(e);continue
            if path.exists():raise RuntimeError("orphan output")
            out,stats=engine.decode(data[row["id"]]);n.guard()
            equality=None
            if not tf32:
                prior=n.load_file(str(ROOT/"outputs/TRR-0020/dev3"/(row["id"]+"__reset_80.0_0.6_0.98.safetensors")))
                equality={k:bool(torch.equal(v,prior[k])) for k,v in out.items()}
                if not all(equality.values()):
                    n.write(X/f"dev5_failed_record_equivalence_{time.time_ns()}.json",{"binding":bound,"id":row["id"],"equality":equality})
                    raise RuntimeError("record graph equivalence failed")
            n.save_file(out,str(path))
            e={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),
               "binding":bound,"stats":stats,"reference_tokens_equal":equality,"frozen_unix":time.time()}
            n.write(rp,e);entries.append(e);print("frozen",row["id"],name,round(stats["total_seconds"],3),flush=True)
        setups[name]["capture_events"]=engine.capture_events
        del engine;gc.collect();torch.cuda.empty_cache()
    if binding()!=bound:raise RuntimeError("source changed")
    assert len(entries)==80
    n.write(X/"dev5_freeze.json",{"task_id":"TRR-0020","binding":bound,"environment":env,"entries":entries,
       "selected_records":selected,"configurations":CONFIGS,"setups":setups,"truth_read":False,
       "start_unix":start,"end_unix":time.time(),"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
       "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
       "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"command":sys.argv})
    print("COMPLETE80CELLFREEZE",flush=True)
if __name__=="__main__":main()

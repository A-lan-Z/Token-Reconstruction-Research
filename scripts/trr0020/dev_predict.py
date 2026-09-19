"""Fixed development matrix; never reads evaluated source text/token labels."""
from pathlib import Path
import sys,json,time,gc,subprocess,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
import native as n
from full_vocabulary import FullVocabularyA2
torch=n.torch
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev1"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
FAMILIES={"embedding":[("white",.5),("white",1.)],
          "intrinsic":[("white",.5),("white",1.),("shifted",1.)],
          "bos":[("white",.5),("white",1.),("shifted",1.)]}
def binding():
    paths=list((ROOT/"scripts/trr0020").glob("*.py"))+[X/"PLAN.md",
       ROOT/"scripts/trr0014/native.py",ROOT/"scripts/agent4/common.py",
       ROOT/"src/token_reconstruction/public_prefix.py",
       ROOT/"src/token_reconstruction/prefix_fragments.py",
       ROOT/"src/token_reconstruction/prefix_weight_metric.py"]
    return {"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
        "observations_sha256":n.digest(INPUT/"observations.safetensors"),
        "metadata_sha256":n.digest(INPUT/"metadata.json"),
        "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors")}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bound=binding();env=n.environment();start=time.time()
    prefix=n.load_prefix()
    data=n.load_file(str(INPUT/"observations.safetensors"))
    selected=[];counts={}
    for r in json.loads((INPUT/"metadata.json").read_text()):
        key=(r["condition"],r["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(r);counts[key]+=1
    assert len(selected)==8
    gen=torch.Generator().manual_seed(200020)
    fixture=torch.randint(256,128000,(1,128),generator=gen).to("cuda");fixture[0,0]=128000
    with torch.inference_mode():qualification_observation=prefix.forward_full(fixture)[0].clone()
    entries=[];setup={}
    for descriptor,rules in FAMILIES.items():
        n.guard();n.sync();t=time.perf_counter()
        engine=FullVocabularyA2(prefix,descriptor)
        n.sync();setup[descriptor]={"seconds":time.perf_counter()-t}
        print("built",descriptor,setup[descriptor],flush=True)
        qualification=[]
        for rule,step in rules:
            name=f"{descriptor}_{rule}_{step}"
            outputs,stats=engine.decode(qualification_observation,rule,step)
            qualification.append({"method":name,"stats":stats,
               "synthetic_correct":{k:int((v[1:]==fixture[0,1:].cpu()).sum()) for k,v in outputs.items()}})
            n.guard()
            print("largest qualified",name,round(stats["total_seconds"],3),
                  qualification[-1]["synthetic_correct"],flush=True)
        free,total=torch.cuda.mem_get_info()
        if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:
            raise RuntimeError("qualification resource margin insufficient")
        n.write(X/(descriptor+"_qualification_"+str(time.time_ns())+".json"),
            {"binding":bound,"environment":env,"rows":qualification,"setup":setup[descriptor],
             "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
             "free_bytes":free})
        for row in selected:
            for rule,step in rules:
                name=f"{descriptor}_{rule}_{step}"
                path=OUT/(row["id"]+"__"+name+".safetensors")
                receipt=path.with_suffix(".json")
                if receipt.exists():
                    item=json.loads(receipt.read_text())
                    if item["binding"]!=bound or n.digest(path)!=item["sha256"]:raise RuntimeError("resume changed")
                    entries.append(item);continue
                if path.exists():raise RuntimeError("orphan output")
                outputs,stats=engine.decode(data[row["id"]],rule,step);n.guard()
                n.save_file(outputs,str(path))
                item={**row,"method":name,"path":str(path.relative_to(ROOT)),
                      "sha256":n.digest(path),"binding":bound,"stats":stats,
                      "frozen_unix":time.time()}
                n.write(receipt,item);entries.append(item)
                print("frozen",row["id"],name,round(stats["total_seconds"],3),flush=True)
        del engine;gc.collect();torch.cuda.empty_cache()
    if bound!=binding():raise RuntimeError("source changed during development")
    assert len(entries)==64
    n.write(X/"dev1_freeze.json",{"task_id":"TRR-0020","binding":bound,"environment":env,
        "selected_records":selected,"families":FAMILIES,"entries":entries,
        "truth_read":False,"start_unix":start,"end_unix":time.time(),"setup":setup,
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
        "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "command":sys.argv})
    print("COMPLETE64CELLFREEZE",flush=True)
if __name__=="__main__":main()

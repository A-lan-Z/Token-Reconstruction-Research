"""Resume frozen dev3/dev4 methods without repeating completed cells.
This changes orchestration only; each prediction calls the original engine.
"""
from pathlib import Path
import argparse,sys,importlib,json,time,subprocess,resource
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument("--stage",type=int,choices=[3,4],required=True)
args=p.parse_args();stage=args.stage
sys.path.insert(0,str(ROOT/f"scripts/trr0020_stage{stage}"))
d=importlib.import_module(f"dev{stage}_predict")
n=d.n;torch=d.torch;X=d.X;OUT=d.OUT;INPUT=d.INPUT
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (X/f"dev{stage}_freeze.json").exists():raise RuntimeError("matrix already frozen")
    bound=d.binding();runner=n.digest(Path(__file__))
    env=n.environment();started=time.time()
    selected=[];counts={}
    for row in json.loads((INPUT/"metadata.json").read_text()):
        key=(row["condition"],row["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(row);counts[key]+=1
    methods=[triple for family in d.FAMILIES.values() for triple in family]
    records={}
    for rp in OUT.glob("*.json"):
        e=json.loads(rp.read_text())
        if e["binding"]!=bound or n.digest(ROOT/e["path"])!=e["sha256"]:
            raise RuntimeError("saved-cell source/output mismatch")
        records[(e["id"],e["method"])]=e
    prefix=n.load_prefix();n.sync();setup_start=time.perf_counter()
    Engine=d.ResetSoftVocabulary if stage==3 else d.DiagonalSoftVocabulary
    engine=Engine(prefix);n.sync();setup_seconds=time.perf_counter()-setup_start
    gen=torch.Generator().manual_seed(200020)
    fixture=torch.randint(256,128000,(1,128),generator=gen).to("cuda");fixture[0,0]=128000
    with torch.inference_mode():h=prefix.forward_full(fixture)[0].clone()
    out,stats=engine.decode(h,80.,.6,.98);n.guard()
    free,total=torch.cuda.mem_get_info()
    if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:
        raise RuntimeError("largest representative cell insufficient margin")
    qpath=X/f"dev{stage}_resume_qualification_{time.time_ns()}.json"
    n.write(qpath,{"binding":bound,"runner_sha256":runner,"environment":env,
       "stats":stats,"synthetic_correct":{k:int((v[1:]==fixture[0,1:].cpu()).sum()) for k,v in out.items()},
       "setup_seconds":setup_seconds,"peak_allocated":torch.cuda.max_memory_allocated(),
       "peak_reserved":torch.cuda.max_memory_reserved(),"free_bytes":free,
       "scope":"fresh largest-geometry check; all original variant qualifications retained"})
    print("Largest representative qualified; resuming",len(records),"saved cells",flush=True)
    data=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for row in selected:
        for scale,lr,decay in methods:
            tag="reset" if stage==3 else "diagonal"
            name=f"{tag}_{scale}_{lr}_{decay}"
            if (row["id"],name) in records:
                entries.append(records[(row["id"],name)]);continue
            path=OUT/(row["id"]+"__"+name+".safetensors");receipt=path.with_suffix(".json")
            if path.exists() or receipt.exists():raise RuntimeError("unverified/orphan existing output")
            n.guard()
            output,stats=engine.decode(data[row["id"]],scale,lr,decay);n.guard()
            n.save_file(output,str(path))
            e={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),
               "binding":bound,"stats":stats,"frozen_unix":time.time(),"resume_runner_sha256":runner}
            n.write(receipt,e);entries.append(e)
            print("frozen",len(entries),len(selected)*len(methods),row["id"],name,
                  round(stats["total_seconds"],3),flush=True)
    if bound!=d.binding() or runner!=n.digest(Path(__file__)):raise RuntimeError("source changed during resume")
    assert len(entries)==len(selected)*len(methods)
    n.write(X/f"dev{stage}_freeze.json",{"task_id":"TRR-0020","binding":bound,
        "environment":env,"selected_records":selected,"families":d.FAMILIES,"entries":entries,
        "truth_read":False,"start_unix":started,"end_unix":time.time(),"setup_seconds":setup_seconds,
        "resume_runner_path":str(Path(__file__).relative_to(ROOT)),"resume_runner_sha256":runner,
        "resumed_existing_cells":len(records),"qualification_path":str(qpath.relative_to(ROOT)),
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
        "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"command":sys.argv,
        "timing_scope":"exploratory; interrupted attempts preserved separately, no comparative latency claim"})
    print("COMPLETE",len(entries),"CELLFREEZE",flush=True)
if __name__=="__main__":main()

from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"));import native as n
sys.path.insert(0,str(ROOT/"scripts/trr0017"));from discrete_parallel import DiscreteParallel
torch=n.torch;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
X=ROOT/"experiments/TRR-0017";OUT=ROOT/"outputs/TRR-0017/dev3"
METHODS=("raw_full_64_0.003","raw_full_128_0.01","white_full_64_0.003",
         "cosine_full_64_0.003","raw_diagonal_64_0.003","cosine_diagonal_64_0.003")
def main():
    OUT.mkdir(parents=True,exist_ok=False);env=n.environment();started=time.time()
    p=n.load_prefix();n.sync();start=time.perf_counter();engine=DiscreteParallel(p);n.sync();setup=time.perf_counter()-start
    n.guard();g=torch.Generator().manual_seed(170017)
    fixture=torch.randint(256,128000,(1,128),generator=g).to("cuda");fixture[0,0]=128000
    h=p.forward_full(fixture)[0].detach().clone();qual=[]
    for m in METHODS:
        out,stats=engine.decode(h,m);n.guard()
        qual.append({"method":m,"stats":stats,"correct":int((out["tokens"][1:]==fixture[0,1:].cpu()).sum())})
        print("qualified",m,round(stats["total"],3),qual[-1]["correct"],flush=True)
    n.write(X/"dev3_qualification.json",{"environment":env,"setup_seconds":setup,"rows":qual,
        "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
    old=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
    meta=json.loads((old/"metadata.json").read_text());obs=n.load_file(str(old/"observations.safetensors"))
    selected=[];counts={}
    for r in meta:
        key=(r["condition"],r["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(r);counts[key]+=1
    entries=[]
    for r in selected:
        for m in METHODS:
            n.guard();out,stats=engine.decode(obs[r["id"]],m)
            path=OUT/(r["id"]+"__"+m+".safetensors");n.save_file(out,str(path))
            entries.append({**r,"method":m,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),**stats})
            print("frozen",r["id"],m,round(stats["total"],3),flush=True)
    n.write(X/"dev3_freeze.json",{"task_id":"TRR-0017","environment":env,"methods":METHODS,
        "input_sha256":n.digest(old/"observations.safetensors"),"metadata_sha256":n.digest(old/"metadata.json"),
        "implementation_sha256":n.digest(ROOT/"scripts/trr0017/discrete_parallel.py"),"entries":entries,
        "truth_read":False,"start_unix":started,"end_unix":time.time(),
        "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
if __name__=="__main__":main()


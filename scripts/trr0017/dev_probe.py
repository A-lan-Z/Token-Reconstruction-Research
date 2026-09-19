"""Preregistered candidate-free development: freeze before opened labels."""
from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
import native as n
sys.path.insert(0,str(ROOT/"scripts/trr0017"))
from sequence_inverse import SequenceInverse
torch=n.torch
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
X=ROOT/"experiments/TRR-0017"; OUT=ROOT/"outputs/TRR-0017/dev1"
METHODS=("reverse32","reverse96","adam_128","adam_384","lbfgs_96")
def write(p,x):n.write(p,x)
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.time();env=n.environment()
    prefix=n.load_prefix();torch.cuda.synchronize();t=time.perf_counter();engine=SequenceInverse(prefix)
    torch.cuda.synchronize();setup=time.perf_counter()-t;n.guard()
    qualification=[]
    # Geometry qualification uses public synthetic IDs, not evaluation labels.
    generator=torch.Generator().manual_seed(170017)
    fixture=torch.randint(256,128000,(1,128),generator=generator,device="cpu").to("cuda");fixture[0,0]=128000
    with torch.no_grad():h=prefix.forward_full(fixture)[0].detach().clone()
    for m in METHODS:
        outputs,stats=engine.decode(h,m)
        qualification.append({"method":m,"stats":stats,
                 "correct":{k:int((v[1:]==fixture[0,1:].cpu()).sum()) for k,v in outputs.items()}})
        n.guard();print("qualified",m,round(stats["total"],3),qualification[-1]["correct"],flush=True)
    write(X/"dev1_qualification.json",{"environment":env,"setup_seconds":setup,"rows":qualification,
          "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
    old=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
    meta=json.loads((old/"metadata.json").read_text());data=n.load_file(str(old/"observations.safetensors"))
    selected=[];counts={}
    for r in meta:
        key=(r["condition"],r["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(r);counts[key]+=1
    assert len(selected)==8
    entries=[]
    for r in selected:
        for m in METHODS:
            n.guard();out,stats=engine.decode(data[r["id"]],m)
            p=OUT/(r["id"]+"__"+m+".safetensors");n.save_file(out,str(p))
            e={**r,"method":m,"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),**stats}
            entries.append(e);print("frozen",r["id"],m,round(stats["total"],3),flush=True)
    write(X/"dev1_freeze.json",{"task_id":"TRR-0017","environment":env,"methods":METHODS,
        "input_sha256":n.digest(old/"observations.safetensors"),"metadata_sha256":n.digest(old/"metadata.json"),
        "implementation_sha256":n.digest(ROOT/"scripts/trr0017/sequence_inverse.py"),
        "qualification_sha256":n.digest(X/"dev1_qualification.json"),
        "entries":entries,"truth_read":False,"start_unix":start,"end_unix":time.time(),
        "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
if __name__=="__main__":main()


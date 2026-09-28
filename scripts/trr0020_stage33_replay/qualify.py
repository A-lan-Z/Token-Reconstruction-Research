"""Fail-closed qualification of all native current-token graph geometries."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
from engine import CurrentVocabulary,n,torch,F,forward,commit
ROOT=Path(__file__).resolve().parents[2];X=ROOT/"experiments/TRR-0020"
OUT=ROOT/"outputs/TRR-0020/dev33_replay";DEST=X/"dev33_replay_qualification.json"
def guard():
    free,_=torch.cuda.mem_get_info()
    available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
    temp=int(subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],text=True).strip())
    if free<3*2**30 or available<8*2**30 or torch.cuda.max_memory_reserved()>8*2**30 or temp>=80:raise RuntimeError("replay resource margin")
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only qualification")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism flags")
    old=json.loads((X/"dev33_public_probe.json").read_text())
    if not old["passed"] or len(old["cells"])!=72:raise RuntimeError("unqualified public rule")
    for name,sha in old["sources"].items():
        if n.digest(ROOT/name)!=sha:raise RuntimeError("public source changed")
    sources=old["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in (ROOT/"scripts/trr0020_stage33_replay").glob("*.py")}
    sources|={str(p.relative_to(ROOT)):n.digest(p) for p in [X/"DEV33_REPLAY_PLAN.md",X/"dev33_replay_preflight.json",X/"dev33_public_probe.json"]}
    result={"task_id":"TRR-0020","kind":"native_context_exact_replay_qualification","sources":sources,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"environment":n.environment(),
      "prefix_sha256":old["prefix_sha256"],"config_sha256":old["config_sha256"],"command":sys.argv,
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"passed":False,"captures":[],"cells":[],"eager":[]}
    def save(label,data):
        path=OUT/(label+".safetensors");t=time.perf_counter();n.save_file({k:v.contiguous() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-t}
    def compare(a,b):
        same=set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items())
        diff={k:float((v.double()-b[k].double()).abs().max()) for k,v in a.items() if k in b and not torch.equal(v,b[k])}
        return same,diff
    try:
        guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        n.sync();result["prefix_preparation_seconds"]=time.perf_counter()-t
        t=time.perf_counter();engine=CurrentVocabulary(prefix);n.sync();result["engine_lookup_preparation_seconds"]=time.perf_counter()-t
        for pos in [127]+list(range(127)):
            engine.ensure(pos);result["captures"].append(engine.capture_events[-1]);guard()
            if pos==127 or pos%32==0:print("NATIVE_CAPTURE",pos,len(engine.graphs),round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        if len(engine.graphs)!=128:raise RuntimeError("incomplete native geometry")
        for length in [128,40]:
            source=next(c["source"] for c in old["contexts"] if c["length"]==length)
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("fixture changed")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda")
            base=prefix.embed_tokens(fixture["public_ids"]);target=fixture["public_target"]
            pe=prefix.rotary_emb(base[None],torch.arange(length,device="cuda")[None]);cos,sin=pe[0][0],pe[1][0]
            selected=[length-1,length//2,1];snapshots={}
            past=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in engine.layers]
            for pos in range(length):
                if pos in selected:snapshots[pos]=past
                _,_,current=forward(base[pos:pos+1],engine.layers,past,cos[pos:pos+1],sin[pos:pos+1]);past=commit(past,current)
            for pos in selected:
                history=snapshots[pos];co,si=cos[pos:pos+1],sin[pos:pos+1];h=target[pos:pos+1]
                for factor in [2.,1.,.5]:
                  for steps in [8,4,2,1]:
                    original=next(c for c in old["cells"] if (c["length"],c["position"],c["factor"],c["steps"])==(length,pos,factor,steps))["repetitions"][0]
                    if n.digest(ROOT/original["path"])!=original["sha256"]:raise RuntimeError("reference changed")
                    reference=n.load_file(str(ROOT/original["path"]));reps=[]
                    result["cells"].append({"length":length,"position":pos,"factor":factor,"steps":steps,"reference":original["path"],
                      "reference_sha256":original["sha256"],"repetitions":reps})
                    for rep in range(3):
                        data,stats=engine.run_current(h,history,co,si,factor,steps,True)
                        equal,diff=compare(data,reference)
                        unchanged=all(torch.equal(pk[:,:pos],k) and torch.equal(pv[:,:pos],v) for (pk,pv),(k,v) in zip(engine.past,history))
                        entry={"rep":rep,"stats":stats,"original_exact":equal,"differences":diff,"past_unchanged":unchanged,
                          **save(f"{length}_{pos}_{factor}_{steps}_{rep}",data)}
                        reps.append(entry)
                        if not equal or not unchanged or not stats["passed"]:raise RuntimeError("replay changed original; saved and excluded")
                        guard()
                    data,stats=engine.run_current(h,history,co,si,factor,steps,False)
                    equal,diff=compare(data,reference)
                    result["eager"].append({"length":length,"position":pos,"factor":factor,"steps":steps,"stats":stats,
                      "original_exact":equal,"differences":diff,**save(f"{length}_{pos}_{factor}_{steps}_eager",data)})
                    if not equal or not stats["passed"]:raise RuntimeError("eager buffer adapter changed original")
                print("CAUSAL_REPLAY_CONTEXT_COMPLETE",length,pos,flush=True)
        if len(result["cells"])!=72 or len(result["eager"])!=72 or not all(len(c["repetitions"])==3 for c in result["cells"]):raise RuntimeError("incomplete replay matrix")
        for name,sha in sources.items():
            if n.digest(ROOT/name)!=sha:raise RuntimeError("source changed")
        result["passed"]=True
    except Exception:
        result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("CAUSAL_REPLAY_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

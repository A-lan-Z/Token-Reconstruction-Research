from pathlib import Path
import sys,json,time,subprocess,os,traceback,resource,argparse,statistics
ROOT=Path(__file__).resolve().parents[2]
for part in ["scripts/trr0020_stage38_grid","scripts/trr0020_stage44","scripts/trr0020_stage45","scripts/trr0020_stage46"]:sys.path.insert(0,str(ROOT/part))
import grid46_support as inherited
sys.path.insert(0,str(Path(__file__).parent))
from history_optimizer import HistoryOptimizer,CONFIGS
from first_optimizer import gradient_reference
n=inherited.n;torch=n.torch;X=inherited.X;guard=inherited.guard
def binding():
    paths=list(Path(__file__).parent.glob("*.py"))+[X/v for v in ["DEV47_R1_PLAN.md","dev47_r1_preflight.json","dev47_r1_cpu_reference.json","dev46_phase_reuse.json"]]
    return {"inherited":inherited.binding(),"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths}}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=list(CONFIGS),required=True);args=parser.parse_args()
    dest=X/f"dev47_r1_public_{args.mode}.json";outdir=ROOT/"outputs/TRR-0020/dev47_r1"/args.mode
    if dest.exists() or outdir.exists():raise RuntimeError("create-only")
    outdir.mkdir(parents=True);bound=binding();guard();started=time.time()
    n.sync();t=time.perf_counter();prefix=n.load_prefix();engine=HistoryOptimizer(prefix,args.mode);n.sync();prep=time.perf_counter()-t
    phase=json.loads((X/"dev46_phase_reuse.json").read_text())
    if n.digest(ROOT/phase["qualification_path"])!=phase["qualification_sha256"]:raise RuntimeError("old qualification")
    old=json.loads((ROOT/phase["qualification_path"]).read_text())
    q={"task_id":"TRR-0020","scope":"known public numerical fixtures only; no benchmark claim","mode":args.mode,"binding":bound,
      "runs":[],"eager":[],"gradients":[],"passed":False,"preparation_seconds":prep,"start_unix":started,"environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":[sys.executable,*sys.argv]}
    ids=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda");ids[0,0]=128000
    def save(label,out):
        path=outdir/(label+".safetensors");t=time.perf_counter();n.save_file(out,str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-t}
    try:
      for length in [128,40]:
        with torch.no_grad():h=prefix.forward_full(ids[:,:length])[0].float()
        refrow=next(v for v in old["runs"] if v["length"]==length and v["steps"]==64 and v["rep"]==0)
        if n.digest(ROOT/refrow["path"])!=refrow["sha256"]:raise RuntimeError("old output hash")
        ref=n.load_file(str(ROOT/refrow["path"]));first=None
        for rep in range(3):
            out,stats=engine.decode(h,64)
            if first is None:first,fs=out,stats
            same=inherited.equal(out,stats,first,fs)
            anchor=inherited.equal(out,stats,ref,refrow["stats"]) if args.mode=="control" else (
              all(inherited.bits_equal(v,ref[k]) for k,v in out.items() if k.startswith("warm_") or k=="initial_embedding") and
              all(inherited.trace_equal(stats["warm"][k],refrow["stats"]["warm"][k]) for k in inherited.inherited.old.TRACE_KEYS))
            q["runs"].append({"length":length,"rep":rep,"stats":stats,"repeat_exact":same,"anchor_exact":anchor,**save(f"public_{length}_{rep}",out)})
            if not same or not anchor or not stats["numeric_valid"]:raise RuntimeError("repeat/anchor failed")
            guard()
        out,stats=engine.decode(h,64,replay=False);same=inherited.equal(out,stats,first,fs)
        q["eager"].append({"length":length,"bytes_equal":same,"stats":stats,**save(f"eager_{length}",out)})
        if not same:raise RuntimeError("eager/replay changed")
        q["gradients"].append({"length":length,**gradient_reference(engine,h)})
        if not q["gradients"][-1]["passed"]:raise RuntimeError("gradient reference")
        if length==128:print("LARGEST_HISTORY_CELL_PASSED",args.mode,round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        guard()
      if binding()!=bound:raise RuntimeError("binding changed")
      q["passed"]=len(q["runs"])==6 and len(q["eager"])==2
    except Exception:q["failure"]=traceback.format_exc();raise
    finally:
      q.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        warm_capture_events=engine.capture_events,mirror_capture_events=engine.mirror_capture_events)
      n.write(dest,q)
      print("HISTORY_PUBLIC_DONE",args.mode,q["passed"],flush=True)
if __name__=="__main__":main()

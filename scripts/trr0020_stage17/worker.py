from support import *
import argparse,subprocess,resource,hashlib
from proximal import reference_test
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True);args=parser.parse_args()
    name,warm,metric,alpha,scale=CONFIGS[args.index];bound=binding();phase=X/f"dev17_phase_{name}.json"
    if phase.exists():
        f=json.loads(phase.read_text())
        if f["binding"]!=bound:raise RuntimeError("phase changed")
        for e in f["entries"]:verify(e,bound)
        return
    OUT.mkdir(parents=True,exist_ok=True);env=n.environment();start=time.time();n.guard()
    reference=reference_test()
    cpu=json.loads((X/"dev17_cpu_reference.json").read_text())
    if not cpu["passed"] or cpu["source_sha256"]!=n.digest(ROOT/"scripts/trr0020_stage17/cpu_reference.py"):raise RuntimeError("unqualified curvature math")
    t=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-t
    t=time.perf_counter();engine=WarmRefinement(prefix,warm,metric,alpha,scale);n.sync();engine_seconds=time.perf_counter()-t
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200041)).to("cuda");fixture[0,0]=128000
    qs=[];eagers=[];cache_checks=[]
    for length in [128,40]:
        with torch.no_grad():h=prefix.forward_full(fixture[:,:length])[0].float()
        first=None;first_stats=None
        for rep in range(3):
            output,stats=engine.decode(h);n.guard()
            stats["probe_sha256"]=hashlib.sha256(engine.direct.probes[:,:length-1].contiguous().cpu().numpy().tobytes()).hexdigest()
            path=X/f"dev17_qualification_{name}_{length}_{rep}_{time.time_ns()}.safetensors";n.save_file(output,str(path))
            if first is None:first=output;first_stats=stats
            free,total=torch.cuda.mem_get_info()
            qs.append({"length":length,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"stats":stats,
              "outputs_and_traces_equal":equal(output,stats,first,first_stats),
              "free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated()})
        cache_checks.append({"length":length,**cache_reference(engine.direct,length)})
        eager,loss=eager_reference(engine.direct,h,first["warm_step"+str(warm)])
        path=X/f"dev17_qualification_eager_{name}_{length}_{time.time_ns()}.safetensors";n.save_file(eager,str(path))
        eagers.append({"length":length,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"loss_trace":loss,
          "tokens_equal":all(torch.equal(eager[k],first[k]) for k in eager),
          "losses_equal":loss==first_stats["direct"]["loss_trace"]})
    q={"binding":bound,"runs":qs,"eager":eagers,"cache_reference":cache_checks,"environment":env,"quadratic_cpu_reference":reference,
       "prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds,"peak_reserved":torch.cuda.max_memory_reserved()}
    n.write(X/f"dev17_qualification_{name}_{time.time_ns()}.json",q)
    if not all(r["stats"]["curvature_numeric_valid"] for r in qs):raise RuntimeError("invalid curvature; evidence saved")
    if not all(r["all_close"] for r in cache_checks):raise RuntimeError("GPU cache reference failed; evidence saved")
    if not all(r["outputs_and_traces_equal"] for r in qs):raise RuntimeError("nonrepeatable; evidence saved")
    if not all(r["tokens_equal"] and r["losses_equal"] for r in eagers):raise RuntimeError("eager control differs; evidence saved")
    if min(r["free_bytes"] for r in qs)<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError("largest memory margin")
    print("QUALIFIED",name,round(qs[-1]["stats"]["total_seconds"],4),round(torch.cuda.max_memory_reserved()/2**30,4),flush=True)
    original={e["id"]:e for e in json.loads((X/"dev7_freeze.json").read_text())["entries"] if e["method"]=="gini003"}
    observed=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for row in rows():
        path=OUT/f'{row["id"]}__{name}.safetensors';rp=path.with_suffix(".json")
        if rp.exists():
            e=json.loads(rp.read_text());verify(e,bound);entries.append(e);continue
        if path.exists():raise RuntimeError("orphan output")
        output,stats=engine.decode(observed[row["id"]]);n.guard();n.save_file(output,str(path))
        previous=original[row["id"]];old_path=ROOT/previous["path"]
        if n.digest(old_path)!=previous["sha256"]:raise RuntimeError("original anchor changed")
        old=n.load_file(str(old_path));keys=["step"+str(s) for s in [0,16,32,64,128] if s<=warm]
        same=all(torch.equal(output["warm_"+k],old[k]) for k in keys) and all(
          stats["soft"][k]==previous["stats"][k][:warm+1] for k in TRACE_KEYS)
        anchor={"path":previous["path"],"sha256":previous["sha256"],"warm_steps":warm,
                "snapshots_and_trace_prefixes_equal":same}
        free,total=torch.cuda.mem_get_info()
        e={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"binding":bound,"stats":stats,
          "original_anchor":anchor,"frozen_unix":time.time(),"free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved()}
        n.write(rp,e)
        if not stats["curvature_numeric_valid"]:raise RuntimeError("invalid cell curvature; saved")
        if not same:raise RuntimeError("archived warm stage differs; saved")
        if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError("cell memory margin")
        entries.append(e);print("FROZEN",row["id"],name,round(stats["total_seconds"],4),flush=True)
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(phase,{"task_id":"TRR-0020","method":name,"configuration":CONFIGS[args.index],"binding":bound,"entries":entries,
      "environment":env,"qualification":q,"capture_events":engine.capture_events,"prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds,
      "start_unix":start,"end_unix":time.time(),"truth_read":False,"peak_reserved":torch.cuda.max_memory_reserved(),
      "peak_allocated":torch.cuda.max_memory_allocated(),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv})
    print("PHASE_COMPLETE",name,flush=True)
if __name__=="__main__":main()

from support import *
import argparse,subprocess,resource,gc
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True);args=parser.parse_args()
    name,mode,lr=CONFIGS[args.index];bound=binding();phase=X/f"dev14_phase_{name}.json"
    if phase.exists():
        f=json.loads(phase.read_text())
        if f["binding"]!=bound:raise RuntimeError("phase changed")
        for e in f["entries"]:verify(e,bound)
        return
    OUT.mkdir(parents=True,exist_ok=True);env=n.environment();start=time.time();n.guard()
    kernel=json.loads((X/"dev14_scaled_kernel_reference.json").read_text())
    if kernel["kernel_source_sha256"]!=n.digest(ROOT/"scripts/trr0020_stage14/epsilon_soft.py"):raise RuntimeError("unqualified kernel source")
    if not all(r["within_declared_tolerance"] and r["finite"] for r in kernel["results"]):raise RuntimeError("failed kernel qualification")
    if not kernel["results"][0]["original_kernel_byte_equal"]:raise RuntimeError("original kernel differs")
    t=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-t
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200040)).to("cuda");fixture[0,0]=128000
    with torch.no_grad():h=prefix.forward_full(fixture)[0].float()
    control=name=="fixed1e12_lr0p6";reference=None;reference_stats=None;paired_reference=None
    if control:
        base=DiscreteSoftVocabulary(prefix,"gini003",.003,0.,32);base.steps=128
        reference,reference_stats=base.decode(h)
        path=X/f"dev14_qualification_original_control_{time.time_ns()}.safetensors";n.save_file(reference,str(path))
        paired_reference={"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"stats":reference_stats}
        del base;gc.collect();torch.cuda.empty_cache()
    t=time.perf_counter();engine=EpsilonVocabulary(prefix,mode,lr);n.sync();engine_seconds=time.perf_counter()-t
    qs=[]
    for rep in range(3):
        output,stats=engine.decode(h);n.guard()
        path=X/f"dev14_qualification_{name}_{rep}_{time.time_ns()}.safetensors";n.save_file(output,str(path))
        if reference is None:reference=output;reference_stats=stats
        free,total=torch.cuda.mem_get_info()
        qs.append({"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"stats":stats,
          "outputs_and_traces_equal":equal(output,stats,reference,reference_stats),
          "free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated()})
    q={"binding":bound,"runs":qs,"paired_original":paired_reference,"environment":env,
       "prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds}
    n.write(X/f"dev14_qualification_{name}_{time.time_ns()}.json",q)
    if not all(r["outputs_and_traces_equal"] for r in qs):raise RuntimeError("nonrepeatable or control changed; evidence saved")
    if min(r["free_bytes"] for r in qs)<2*2**30 or max(r["peak_reserved"] for r in qs)>6*2**30:raise RuntimeError("largest memory margin")
    print("QUALIFIED",name,round(qs[-1]["stats"]["total_seconds"],4),round(torch.cuda.max_memory_reserved()/2**30,4),flush=True)
    old=json.loads((X/"dev7_freeze.json").read_text());original={e["id"]:e for e in old["entries"] if e["method"]=="gini003"}
    observed=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for row in rows():
        path=OUT/f'{row["id"]}__{name}.safetensors';rp=path.with_suffix(".json")
        if rp.exists():
            e=json.loads(rp.read_text());verify(e,bound);entries.append(e);continue
        if path.exists():raise RuntimeError("orphan output")
        output,stats=engine.decode(observed[row["id"]]);n.guard();n.save_file(output,str(path))
        anchor=None
        if control:
            previous=original[row["id"]];old_path=ROOT/previous["path"]
            if n.digest(old_path)!=previous["sha256"]:raise RuntimeError("original anchor changed")
            same=equal(output,stats,n.load_file(str(old_path)),previous["stats"],["step0","step16","step32","step64","step128"])
            anchor={"path":previous["path"],"sha256":previous["sha256"],"snapshots_and_trace_prefixes_equal":same}
        free,total=torch.cuda.mem_get_info()
        e={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"binding":bound,"stats":stats,
          "original_anchor":anchor,"frozen_unix":time.time(),"free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved()}
        n.write(rp,e)
        if anchor and not anchor["snapshots_and_trace_prefixes_equal"]:raise RuntimeError("archived control differs; saved")
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

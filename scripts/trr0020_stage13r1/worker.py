from support import *
import argparse,subprocess,resource
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True);args=parser.parse_args()
    name,strength,lr=CONFIGS[args.index];bound=binding();phase=X/f"dev13_r1_phase_{name}.json"
    if phase.exists():
        f=json.loads(phase.read_text())
        if f["binding"]!=bound:raise RuntimeError("changed phase")
        for e in f["entries"]:verify(e,bound)
        print("VERIFIED_EXISTING_PHASE",name,flush=True);return
    OUT.mkdir(parents=True,exist_ok=True);start=time.time();env=n.environment();n.guard()
    original.reference_tests()
    t=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-t
    t=time.perf_counter();engine=Engine(prefix,strength,lr);n.sync();engine_seconds=time.perf_counter()-t
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200033)).to("cuda");fixture[0,0]=128000
    with torch.no_grad():h=prefix.forward_full(fixture)[0].float()
    prior_paths=list(X.glob(f"dev13_repeatability_{name}_*.json"))
    if len(prior_paths)!=1:raise RuntimeError("expected original public qualification")
    prior=json.loads(prior_paths[0].read_text())
    if prior["binding"]!=original.binding():raise RuntimeError("original binding changed")
    reference=n.load_file(str(ROOT/prior["runs"][0]["path"]))
    if n.digest(ROOT/prior["runs"][0]["path"])!=prior["runs"][0]["sha256"]:raise RuntimeError("original public output changed")
    qualification=[]
    for rep in range(3):
        data,stats=engine.decode(h);n.guard()
        raw=X/f"dev13_r1_qualification_{name}_{rep}_{time.time_ns()}.safetensors";n.save_file(data,str(raw))
        free,total=torch.cuda.mem_get_info()
        qualification.append({"path":str(raw.relative_to(ROOT)),"sha256":n.digest(raw),"stats":stats,
          "equal_to_original":same_numerics(data,stats,reference,prior["runs"][0]["stats"]),
          "free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated()})
    receipt={"binding":bound,"environment":env,"runs":qualification,"prior_public_receipt":str(prior_paths[0].relative_to(ROOT)),
      "prior_public_receipt_sha256":n.digest(prior_paths[0]),"prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds}
    n.write(X/f"dev13_r1_qualification_{name}_{time.time_ns()}.json",receipt)
    if not all(q["equal_to_original"] for q in qualification):raise RuntimeError("isolation changes numerical result; saved and excluded")
    if min(q["free_bytes"] for q in qualification)<2*2**30 or max(q["peak_reserved"] for q in qualification)>6*2**30:
        raise RuntimeError("isolated largest case exceeds margin")
    print("QUALIFIED_IDENTICAL",name,round(torch.cuda.max_memory_reserved()/2**30,4),flush=True)
    observed=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for row in selected_rows():
        path=OUT/f'{row["id"]}__{name}.safetensors';rp=path.with_suffix(".json")
        if rp.exists():
            e=json.loads(rp.read_text());verify(e,bound);entries.append(e);continue
        if path.exists():raise RuntimeError("orphan output")
        data,stats=engine.decode(observed[row["id"]]);n.guard()
        n.save_file(data,str(path))
        free,total=torch.cuda.mem_get_info()
        old_path=ROOT/"outputs/TRR-0020/dev13"/path.name;old_receipt=old_path.with_suffix(".json")
        anchor=None
        if old_receipt.exists():
            old=json.loads(old_receipt.read_text())
            if old["binding"]!=original.binding() or n.digest(old_path)!=old["sha256"]:raise RuntimeError("original cell changed")
            old_data=n.load_file(str(old_path));equal=same_numerics(data,stats,old_data,old["stats"])
            anchor={"path":str(old_receipt.relative_to(ROOT)),"sha256":n.digest(old_receipt),"all_outputs_and_traces_equal":equal}
        e={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"binding":bound,"stats":stats,
          "original_anchor":anchor,"frozen_unix":time.time(),"free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved()}
        n.write(rp,e)
        if anchor is not None and not anchor["all_outputs_and_traces_equal"]:raise RuntimeError("isolated cell differs; saved and excluded")
        if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError("cell exceeds memory margin")
        entries.append(e);print("FROZEN",row["id"],name,round(stats["total_seconds"],4),flush=True)
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(phase,{"task_id":"TRR-0020","binding":bound,"method":name,"configuration":CONFIGS[args.index],"entries":entries,
      "environment":env,"qualification":receipt,"capture_events":engine.capture_events,"prefix_seconds":prefix_seconds,
      "engine_seconds":engine_seconds,"start_unix":start,"end_unix":time.time(),"truth_read":False,
      "peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated(),
      "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv})
    print("PHASE_COMPLETE",name,flush=True)
if __name__=="__main__":main()

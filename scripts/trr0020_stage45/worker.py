from grid45_support import *
import argparse,resource,traceback,statistics
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["control","exact"],required=True);args=parser.parse_args()
    control=args.mode=="control";bound=binding();phase_path=X/f"dev45_phase_{args.mode}.json"
    if phase_path.exists():
        phase=json.loads(phase_path.read_text())
        if phase["binding"]!=bound:raise RuntimeError("resume phase changed")
        for entry in phase["entries"]:verify(entry,bound)
        return
    OUT.mkdir(parents=True,exist_ok=True);started=time.time();env=n.environment();guard()
    kernel=json.loads((X/"dev44_gpu_check.json").read_text())
    if not kernel["passed"]:raise RuntimeError("unqualified kernel")
    n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-t
    t=time.perf_counter();engine=ExactOptimizer(prefix,not control);n.sync();engine_seconds=time.perf_counter()-t
    old_phase=json.loads((X/"dev43_phase_control.json").read_text())
    if n.digest(ROOT/old_phase["qualification_path"])!=old_phase["qualification_sha256"]:raise RuntimeError("old qualification changed")
    old=json.loads((ROOT/old_phase["qualification_path"]).read_text())
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda");fixture[0,0]=128000
    q={"task_id":"TRR-0020","binding":bound,"mode":args.mode,"runs":[],"eager":[],"gradients":[],"passed":False}
    qp=X/f"dev45_qualification_{args.mode}_{time.time_ns()}.json"
    def save(label,out):
        p=OUT/(label+".safetensors");t=time.perf_counter();n.save_file(out,str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
      for length in [128,40]:
        with torch.no_grad():h=prefix.forward_full(fixture[:,:length])[0].float()
        for steps in STEPS:
            old_row=next(v for v in old["runs"] if v["length"]==length and v["steps"]==steps and v["rep"]==0)
            if n.digest(ROOT/old_row["path"])!=old_row["sha256"]:raise RuntimeError("old public output changed")
            ref=n.load_file(str(ROOT/old_row["path"]))
            first=None
            for rep in range(3):
                out,stats=engine.decode(h,steps)
                if first is None:first,first_stats=out,stats
                same=equal(out,stats,first,first_stats);anchored=anchor(out,stats,ref,old_row["stats"],steps,control)
                entry={"length":length,"steps":steps,"rep":rep,"stats":stats,"exact_repeat":same,"anchor_valid":anchored,
                  **save(f"public_{args.mode}_{length}_{steps}_{rep}_{time.time_ns()}",out)}
                q["runs"].append(entry)
                if not same or not anchored or not stats["numeric_valid"]:raise RuntimeError("public repeat/anchor")
                guard()
            out,stats=engine.decode(h,steps,replay=False)
            same=equal(out,stats,first,first_stats)
            q["eager"].append({"length":length,"steps":steps,"exact_equal":same,"stats":stats,
              **save(f"eager_{args.mode}_{length}_{steps}_{time.time_ns()}",out)})
            if not same:raise RuntimeError("eager/replay mismatch")
            guard()
            if length==128 and steps==128:
                q["largest_cell"]={"passed":True,"peak_reserved":torch.cuda.max_memory_reserved()}
                print("LARGEST_EXACT_DECODER_PASSED",args.mode,round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        q["gradients"].append({"length":length,**gradient_reference(engine,h)})
        if not q["gradients"][-1]["passed"]:raise RuntimeError("gradient reference")
      q["passed"]=len(q["runs"])==12 and len(q["eager"])==4 and len(q["gradients"])==2
    except Exception:q["failure"]=traceback.format_exc();raise
    finally:
      q.update(peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated())
      n.write(qp,q)
    old_entries={(e["id"],e["steps"]):e for e in json.loads((X/"dev43_freeze.json").read_text())["entries"] if e["mode"]=="control"}
    observations=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for steps in STEPS:
      name=f"{args.mode}_steps{steps}"
      for row in rows():
        receipt=OUT/f"{row['id']}__{name}.json"
        if receipt.exists():
            entry=json.loads(receipt.read_text());verify(entry,bound);entries.append(entry);continue
        old_entry=old_entries[row["id"],steps]
        if n.digest(ROOT/old_entry["path"])!=old_entry["sha256"]:raise RuntimeError("old real anchor changed")
        ref=n.load_file(str(ROOT/old_entry["path"]));first=None;repetitions=[]
        for rep in range(3):
            out,stats=engine.decode(observations[row["id"]],steps)
            raw=save(f"{row['id']}__{name}__{rep}",out)
            if first is None:first,first_stats=out,stats
            same=equal(out,stats,first,first_stats);anchored=anchor(out,stats,ref,old_entry["stats"],steps,control)
            repetitions.append({"rep":rep,"stats":stats,"exact_repeat":same,"anchor_valid":anchored,**raw})
            if not same or not anchored or not stats["numeric_valid"]:raise RuntimeError("real repeat/anchor; preserved unscored")
            guard()
        entry={**row,"method":name,"mode":args.mode,"steps":steps,"binding":bound,"repetitions":repetitions,
          **{k:repetitions[0][k] for k in ["path","sha256"]},"stats":first_stats,
          "median_seconds":statistics.median(v["stats"]["total_seconds"] for v in repetitions),
          "old_anchor":{"path":old_entry["path"],"sha256":old_entry["sha256"]},"frozen_unix":time.time(),
          "peak_reserved":torch.cuda.max_memory_reserved()}
        n.write(receipt,entry);entries.append(entry)
        print("EXACT_DECODER_FROZEN",name,row["id"],round(entry["median_seconds"],4),flush=True)
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(phase_path,{"task_id":"TRR-0020","mode":args.mode,"binding":bound,"entries":entries,
      "qualification_path":str(qp.relative_to(ROOT)),"qualification_sha256":n.digest(qp),"environment":env,
      "prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds,
      "warm_capture_events":engine.capture_events,"mirror_capture_events":engine.mirror_capture_events,
      "peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated(),
      "peak_host_rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"start_unix":started,"end_unix":time.time(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,"truth_read":False})
if __name__=="__main__":main()

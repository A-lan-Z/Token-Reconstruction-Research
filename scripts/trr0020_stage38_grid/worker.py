from support import *
import argparse,resource,traceback,statistics
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["control","locked"],required=True);args=parser.parse_args()
    locked=args.mode=="locked";bound=binding();phase_path=X/f"dev38_grid_phase_{args.mode}.json"
    if phase_path.exists():
        prior=json.loads(phase_path.read_text())
        if prior["binding"]!=bound:raise RuntimeError("phase changed")
        for entry in prior["entries"]:verify(entry,bound)
        return
    OUT.mkdir(parents=True,exist_ok=True);started=time.time();env=n.environment();guard()
    public=json.loads((X/"dev38_public_probe.json").read_text())
    if not public["passed"] or len(public["families"])!=4:raise RuntimeError("lookup not qualified")
    if n.digest(Path(bound["table_path"]))!=bound["table_sha256"]:raise RuntimeError("derived table changed")
    n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-t
    t=time.perf_counter();table=n.load_file(bound["table_path"],device="cuda")["response"] if locked else None;n.sync()
    table_load_seconds=time.perf_counter()-t
    t=time.perf_counter();engine=FirstOptimizer(prefix,locked,table);n.sync();engine_seconds=time.perf_counter()-t
    old_phase=json.loads((X/"dev32_grid_phase_cold_factor2.0_iter4.json").read_text())
    if n.digest(ROOT/old_phase["qualification_path"])!=old_phase["qualification_sha256"]:raise RuntimeError("old qualification changed")
    old_public=json.loads((ROOT/old_phase["qualification_path"]).read_text())
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda");fixture[0,0]=128000
    q={"task_id":"TRR-0020","binding":bound,"mode":args.mode,"runs":[],"eager":[],"gradients":[],"anchors":[],"passed":False}
    qp=X/f"dev38_grid_qualification_{args.mode}_{time.time_ns()}.json"
    def save(label,output):
        path=OUT/(label+".safetensors");t=time.perf_counter();n.save_file(output,str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-t}
    try:
        for length in [128,40]:
            h=prefix.forward_full(fixture[:,:length])[0].float()
            old_row=next(v for v in old_public["runs"] if v["length"]==length and v["repetition"]==0)
            if n.digest(ROOT/old_row["path"])!=old_row["sha256"]:raise RuntimeError("old public output changed")
            ref=n.load_file(str(ROOT/old_row["path"]))
            for steps in STEPS:
                first=None
                for rep in range(3):
                    output,stats=engine.decode(h,steps)
                    entry={"length":length,"steps":steps,"rep":rep,"stats":stats,**save(f"public_{args.mode}_{length}_{steps}_{rep}_{time.time_ns()}",output)}
                    if first is None:first,first_stats=output,stats
                    entry["exact_repeat"]=equal(output,stats,first,first_stats)
                    entry["fixed_position_valid"]=check_fixed(output,engine,locked)
                    anchor=warm_anchor(output,stats,ref,old_row["stats"],steps,locked)
                    entry["anchor_valid"]=anchor;q["runs"].append(entry)
                    q["anchors"].append({"length":length,"steps":steps,"rep":rep,"path":old_row["path"],"sha256":old_row["sha256"],"passed":anchor})
                    if not entry["exact_repeat"] or not anchor or not entry["fixed_position_valid"] or not stats["numeric_valid"]:
                        raise RuntimeError("public qualification failed; output preserved")
                    guard()
                output,stats=engine.decode(h,steps,replay=False)
                entry={"length":length,"steps":steps,"stats":stats,**save(f"eager_{args.mode}_{length}_{steps}_{time.time_ns()}",output),
                  "exact_equal":equal(output,stats,first,first_stats),"fixed_position_valid":check_fixed(output,engine,locked)}
                q["eager"].append(entry)
                if not entry["exact_equal"] or not entry["fixed_position_valid"]:raise RuntimeError("eager/replay disagreement")
                guard()
                if length==128 and steps==64:
                    q["largest_cell"]={"length":128,"steps":64,"peak_reserved":torch.cuda.max_memory_reserved(),"passed":True}
                    print("LARGEST_FIRST_DECODER_QUALIFIED",args.mode,round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
            q["gradients"].append({"length":length,**gradient_reference(engine,h)})
            if not q["gradients"][-1]["passed"]:raise RuntimeError("independent gradient disagreement")
        q["passed"]=len(q["runs"])==18 and len(q["eager"])==6 and len(q["gradients"])==2
    except Exception:q["failure"]=traceback.format_exc();raise
    finally:
        q.update(peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          prefix_seconds=prefix_seconds,table_load_seconds=table_load_seconds,engine_seconds=engine_seconds)
        n.write(qp,q)
    if not q["passed"]:raise RuntimeError("incomplete qualification")
    print("FIRST_DECODER_PUBLIC_QUALIFIED",args.mode,flush=True)
    anchors={e["id"]:e for e in json.loads((X/"dev32_grid_freeze.json").read_text())["entries"] if e["method"]=="cold_factor2.0_iter4"}
    observations=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for steps in STEPS:
        name=f"{args.mode}_steps{steps}"
        for row in rows():
            receipt=OUT/f"{row['id']}__{name}.json"
            if receipt.exists():
                entry=json.loads(receipt.read_text());verify(entry,bound);entries.append(entry);continue
            first=None;repetitions=[];anchor=anchors[row["id"]]
            if n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise RuntimeError("old real control changed")
            old_output=n.load_file(str(ROOT/anchor["path"]))
            for rep in range(3):
                output,stats=engine.decode(observations[row["id"]],steps)
                raw=save(f"{row['id']}__{name}__{rep}",output)
                if first is None:first,first_stats=output,stats
                valid=equal(output,stats,first,first_stats)
                fixed=check_fixed(output,engine,locked)
                anchored=warm_anchor(output,stats,old_output,anchor["stats"],steps,locked)
                repetitions.append({"rep":rep,"stats":stats,"exact_repeat":valid,"fixed_position_valid":fixed,"anchor_valid":anchored,**raw})
                if not valid or not fixed or not anchored or not stats["numeric_valid"]:raise RuntimeError("invalid real cell; preserved unscored")
                guard()
            entry={**row,"method":name,"mode":args.mode,"steps":steps,"binding":bound,"repetitions":repetitions,
              **{k:repetitions[0][k] for k in ["path","sha256"]},"stats":first_stats,
              "median_seconds":statistics.median(v["stats"]["total_seconds"] for v in repetitions),
              "old_anchor":{"path":anchor["path"],"sha256":anchor["sha256"]},"frozen_unix":time.time(),
              "peak_reserved":torch.cuda.max_memory_reserved()}
            n.write(receipt,entry);entries.append(entry)
            print("FIRST_DECODER_FROZEN",name,row["id"],round(entry["median_seconds"],4),flush=True)
    if binding()!=bound:raise RuntimeError("binding changed")
    n.write(phase_path,{"task_id":"TRR-0020","mode":args.mode,"binding":bound,"entries":entries,
      "qualification_path":str(qp.relative_to(ROOT)),"qualification_sha256":n.digest(qp),"environment":env,
      "prefix_seconds":prefix_seconds,"table_load_seconds":table_load_seconds,"engine_seconds":engine_seconds,
      "warm_capture_events":engine.capture_events,"mirror_capture_events":engine.mirror_capture_events,
      "peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated(),
      "peak_host_rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"start_unix":started,"end_unix":time.time(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,"truth_read":False})
    print("FIRST_DECODER_PHASE_COMPLETE",args.mode,flush=True)
if __name__=="__main__":main()

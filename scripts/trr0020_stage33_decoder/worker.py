from support import *
import argparse,traceback,resource
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--factor",type=float,required=True);args=parser.parse_args()
    factor=args.factor
    if factor not in [2.,1.,.5]:raise ValueError("unknown factor")
    methods=[c for c in CONFIGS if c[1]==factor];bound=binding();phase=X/f"dev33_decoder_phase_factor{factor}.json"
    if phase.exists():
        old=json.loads(phase.read_text())
        if old["binding"]!=bound:raise RuntimeError("changed completed phase")
        for row in old["entries"]:verify(row,bound)
        return
    OUT.mkdir(parents=True,exist_ok=True);start=time.time();guard()
    n.sync();t=time.perf_counter()
    prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
    n.sync();prefix_seconds=time.perf_counter()-t
    t=time.perf_counter();engine=CausalDecoder(prefix);n.sync();engine_seconds=time.perf_counter()-t
    capture_seconds=engine.prepare_all(guard)
    public=json.loads((X/"dev33_public_probe.json").read_text())
    q={"binding":bound,"factor":factor,"runs":[],"eager":[],"history_references":[],"passed":False,
      "prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds,"capture_seconds":capture_seconds,"capture_events":engine.capture_events}
    qp=X/f"dev33_decoder_qualification_factor{factor}_{time.time_ns()}.json"
    def save(label,data):
        p=OUT/(label+".safetensors");t=time.perf_counter();n.save_file(data,str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
        for length in [128,40]:
            source=next(v["source"] for v in public["contexts"] if v["length"]==length)
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("public fixture changed")
            fixture=n.load_file(str(ROOT/source["path"]));h=fixture["public_target"]
            for name,_,steps in methods:
                first=None
                for rep in range(3):
                    output,stats=engine.decode(h,factor,steps,True)
                    if first is None:first=output
                    q["runs"].append({"length":length,"method":name,"steps":steps,"rep":rep,
                      "exact_repeat":equal(first,output),"stats":stats,**save(f"public_{length}_{name}_{rep}",output)})
                    guard()
                q["history_references"].append({"length":length,"method":name,**engine.history_reference(output["tokens"])})
                eager,stats=engine.decode(h,factor,steps,False)
                q["eager"].append({"length":length,"method":name,"steps":steps,"exact_repeat":equal(first,eager),
                  "stats":stats,**save(f"public_{length}_{name}_eager",eager)})
                guard()
        q["passed"]=(len(q["runs"])==24 and len(q["eager"])==8 and len(q["history_references"])==8
          and all(v["exact_repeat"] and v["stats"]["numeric_valid"] for v in q["runs"]+q["eager"])
          and all(v["passed"] for v in q["history_references"]))
        if not q["passed"]:raise RuntimeError("whole-decoder qualification failed; preserved")
    except Exception:q["failure"]=traceback.format_exc();raise
    finally:
        q.update(peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated());n.write(qp,q)
    print("CAUSAL_DECODER_QUALIFIED",factor,round(q["peak_reserved"]/2**30,3),flush=True)
    observed=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for name,_,steps in methods:
      for row in rows():
        p=OUT/f"{row['id']}__{name}.safetensors";receipt=p.with_suffix(".json")
        if receipt.exists():
            entry=json.loads(receipt.read_text());verify(entry,bound);entries.append(entry);continue
        if p.exists():raise RuntimeError("orphan output")
        output,stats=engine.decode(observed[row["id"]],factor,steps,True)
        t=time.perf_counter();n.save_file(output,str(p));sha=n.digest(p);io=time.perf_counter()-t
        entry={**row,"method":name,"binding":bound,"stats":stats,"path":str(p.relative_to(ROOT)),"sha256":sha,
          "frozen_unix":time.time(),"io_hash_seconds":io,"peak_reserved":torch.cuda.max_memory_reserved()}
        n.write(receipt,entry)
        if not stats["numeric_valid"]:raise RuntimeError("nonfinite output")
        entries.append(entry);guard()
        print("FROZEN",row["id"],name,round(stats["total_seconds"],4),flush=True)
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(phase,{"task_id":"TRR-0020","binding":bound,"factor":factor,"methods":methods,"entries":entries,
      "qualification_path":str(qp.relative_to(ROOT)),"qualification_sha256":n.digest(qp),"environment":n.environment(),
      "start_unix":start,"end_unix":time.time(),"truth_read":False,"prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds,
      "capture_seconds":capture_seconds,"peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated(),
      "peak_host_rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv})
    print("CAUSAL_FACTOR_COMPLETE",factor,flush=True)
if __name__=="__main__":main()

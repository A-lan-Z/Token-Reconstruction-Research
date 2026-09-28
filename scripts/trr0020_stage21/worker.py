from support import *
import argparse,resource,traceback
from linearized_prefix import forward,diagonal_jvp,krylov_least_squares
from krylov_graph import direction
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True);args=parser.parse_args()
    name,warm,k,damping=CONFIGS[args.index];bound=binding();phase=X/f"dev21_phase_{name}.json"
    if phase.exists():
        old=json.loads(phase.read_text())
        if old["binding"]!=bound:raise RuntimeError("changed phase")
        for entry in old["entries"]:verify(entry,bound)
        return
    OUT.mkdir(parents=True,exist_ok=True);env=n.environment();started=time.time();guard()
    cpu=json.loads((X/"dev21_graph_solver_cpu.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("unqualified linear solver")
    for p,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/p)!=sha:raise RuntimeError("changed CPU solver source")
    prior=json.loads((X/"dev21_gpu_qualification.json").read_text())
    if not prior["passed"] or prior["sources"]["scripts/trr0020_stage21/linearized_prefix.py"]!=n.digest(ROOT/"scripts/trr0020_stage21/linearized_prefix.py"):
        raise RuntimeError("unqualified real-prefix derivatives")
    n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-t
    t=time.perf_counter();engine=LocalInverse(prefix,warm,k,damping);n.sync();engine_seconds=time.perf_counter()-t
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda");fixture[0,0]=128000
    q={"binding":bound,"environment":env,"method":name,"runs":[],"eager":[],"score_reference":[],"linear_reference":[],"passed":False}
    qp=X/f"dev21_qualification_{name}_{time.time_ns()}.json"
    def save_public(label,output):
        path=OUT/f"qualification_{name}_{label}_{time.time_ns()}.safetensors";n.save_file(output,str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path)}
    try:
        for length in [128,40]:
            h=prefix.forward_full(fixture[:,:length])[0].float();first=None
            for rep in range(3):
                output,stats=engine.decode(h)
                entry={"length":length,"repetition":rep,**save_public(f"{length}_{rep}",output),"stats":stats}
                if first is None:first,first_stats=output,stats
                free,_=torch.cuda.mem_get_info()
                entry.update(outputs_and_traces_equal=equal(output,stats,first,first_stats),free_bytes=free,peak_reserved=torch.cuda.max_memory_reserved())
                q["runs"].append(entry);guard()
            ref,rs=engine.decode(h,replay=False)
            q["eager"].append({"length":length,**save_public(f"{length}_eager",ref),"stats":rs,"outputs_and_traces_equal":equal(first,first_stats,ref,rs)})
            q["score_reference"].append({"length":length,**score_reference(engine,first["embedding_final"][1:])})
            with torch.no_grad():
                cos,sin=engine.local_geometry[length]
                pred,cache=forward(engine.x[:length],engine.layers,cos,sin)
                rhs=engine.target[:length]-pred;rhs[0]=0
                mv=lambda value:diagonal_jvp(value,engine.layers,cache,cos,sin)
                fast,projected,info=direction(mv,rhs,k)
                old,os=krylov_least_squares(mv,rhs,k)
                check={"length":length,**save_public(f"{length}_linear",{"fast":fast,"old":old,"projected_residual":projected,"true_residual":os["residual_norm"]}),
                  "direction_equal":torch.equal(fast,old),"solver_info_zero":bool((info==0).all()),
                  "residual_close":bool(torch.allclose(projected,os["residual_norm"],rtol=1e-3,atol=5e-5))}
                q["linear_reference"].append(check)
            guard()
        q.update(peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),prefix_seconds=prefix_seconds,engine_seconds=engine_seconds)
        q["passed"]=(all(v["outputs_and_traces_equal"] and v["stats"]["numeric_valid"] for v in q["runs"]+q["eager"])
          and all(v["all_close"] for v in q["score_reference"])
          and all(v["direction_equal"] and v["solver_info_zero"] and v["residual_close"] for v in q["linear_reference"]))
        if not q["passed"]:raise RuntimeError("public numerical/equivalence qualification failed; outputs preserved")
    except Exception:
        q["failure"]=traceback.format_exc()
        raise
    finally:
        q.update(peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated())
        n.write(qp,q)
    print("QUALIFIED",name,round(q["peak_reserved"]/2**30,4),flush=True)
    anchors={e["id"]:e for e in json.loads((X/"dev7_freeze.json").read_text())["entries"] if e["method"]=="gini003"}
    means={e["id"]:e for e in json.loads((X/"dev19_freeze.json").read_text())["entries"] if e["method"]=="warm32"}
    observed=n.load_file(str(INPUT/"observations.safetensors"));entries=[]
    for row in rows():
        path=OUT/f"{row['id']}__{name}.safetensors";receipt=path.with_suffix(".json")
        if receipt.exists():
            entry=json.loads(receipt.read_text());verify(entry,bound);entries.append(entry);continue
        if path.exists():raise RuntimeError("orphan output")
        output,stats=engine.decode(observed[row["id"]]);n.save_file(output,str(path))
        anchor=anchors[row["id"]]
        if n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise RuntimeError("changed warm anchor")
        old=n.load_file(str(ROOT/anchor["path"]))
        keys=["step"+str(v) for v in [0,16,32] if v<=warm]
        same=all(torch.equal(output["warm_"+key],old[key]) for key in keys) and all(stats["warm"][key]==anchor["stats"][key][:warm+1] for key in TRACE_KEYS)
        initial_mean=None
        if warm==32:
            original=means[row["id"]]
            if n.digest(ROOT/original["path"])!=original["sha256"]:raise RuntimeError("changed mean anchor")
            exact=n.load_file(str(ROOT/original["path"]))["fitted_embedding"]
            initial_mean={"path":original["path"],"sha256":original["sha256"],"equal":torch.equal(exact,output["initial_embedding"])}
        free,_=torch.cuda.mem_get_info()
        entry={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"binding":bound,
          "stats":stats,"frozen_unix":time.time(),"peak_reserved":torch.cuda.max_memory_reserved(),"free_bytes":free,
          "warm_anchor":{"path":anchor["path"],"sha256":anchor["sha256"],"equal":same},"initial_mean_anchor":initial_mean}
        n.write(receipt,entry)
        if not same or not stats["numeric_valid"] or initial_mean is not None and not initial_mean["equal"]:
            raise RuntimeError("invalid cell or warm anchor; output preserved")
        guard();entries.append(entry)
        print("FROZEN",row["id"],name,round(stats["total_seconds"],4),flush=True)
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(phase,{"task_id":"TRR-0020","binding":bound,"method":name,"configuration":CONFIGS[args.index],"entries":entries,
      "qualification_path":str(qp.relative_to(ROOT)),"qualification_sha256":n.digest(qp),"environment":env,
      "start_unix":started,"end_unix":time.time(),"truth_read":False,"prefix_seconds":prefix_seconds,"engine_seconds":engine_seconds,
      "warm_capture_events":engine.capture_events,"local_capture_events":engine.local_capture_events,
      "peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated(),
      "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv})
    print("PHASE_COMPLETE",name,flush=True)
if __name__=="__main__":main()

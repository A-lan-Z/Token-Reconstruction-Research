"""Public execution profile for unchanged current full-vocabulary KL optimizer."""
from pathlib import Path
import sys,os,time,json,subprocess,resource,traceback,statistics,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage38_grid"))
import support as inherited
from budget_step import update
n=inherited.n;torch=n.torch;FirstOptimizer=inherited.FirstOptimizer;guard=inherited.guard
parser=argparse.ArgumentParser();parser.add_argument("--length",type=int,choices=[128,40],required=True);ARGS=parser.parse_args()
X=ROOT/"experiments/TRR-0020";BASE=f"dev41_profile_r2_{ARGS.length}";OUT=ROOT/"outputs/TRR-0020"/BASE;DEST=X/(BASE+".json")
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism")
    bound=inherited.binding();old_phase=json.loads((X/"dev38_grid_phase_control.json").read_text())
    if n.digest(ROOT/old_phase["qualification_path"])!=old_phase["qualification_sha256"]:raise RuntimeError("old qualification changed")
    old=json.loads((ROOT/old_phase["qualification_path"]).read_text())
    paths=list(Path(__file__).parent.glob("*.py"))+[X/v for v in ["DEV41_PROFILE_R2_PLAN.md","dev41_profile_r2_preflight.json"]]
    result={"task_id":"TRR-0020","scope":"public operator diagnosis; no new reconstruction method or benchmark accuracy",
      "binding":bound,"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"environment":n.environment(),"start_unix":time.time(),
      "truth_read":False,"shortlist":None,"passed":False,"runs":[],"profiles":[],"update_fixtures":[]}
    def save(name,values):
        p=OUT/(name+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in values.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    def profile(name,fn):
        n.sync();t=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            for _ in range(5):fn()
            n.sync()
        seconds=time.perf_counter()-t
        p=OUT/(name+"_trace.json");prof.export_chrome_trace(str(p))
        rows=[]
        for event in prof.key_averages(group_by_input_shape=True):
            device=getattr(event,"self_device_time_total",0.)
            if device:rows.append({"operation":event.key,"count":event.count,"self_gpu_us":device,"input_shapes":event.input_shapes})
        rows.sort(key=lambda v:-v["self_gpu_us"])
        raw_trace=json.loads(p.read_text())
        gpu_events=[v for v in raw_trace["traceEvents"] if v.get("ph")=="X" and v.get("cat") in ["kernel","gpu_memcpy","gpu_memset"]]
        kernel_us=sum(v["dur"] for v in gpu_events)
        if kernel_us<=0:raise RuntimeError("missing GPU activity events")
        return {"name":name,"iterations":5,"profile_wall_seconds":seconds,
          "self_gpu_microseconds_sum":kernel_us,"gpu_activity_event_count":len(gpu_events),"gpu_time_source":"raw CUDA kernel/memcpy/memset events; excludes duplicate CPU attribution","rows":rows,
          "trace":{"path":str(p.relative_to(ROOT)),"sha256":n.digest(p)},
          "table":prof.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total",row_limit=30)}
    try:
      guard()
      if torch.cuda.mem_get_info()[0]<11*2**30:raise RuntimeError("profile needs11GiBfree pre-load margin")
      n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();result["prefix_seconds"]=time.perf_counter()-t
      t=time.perf_counter();engine=FirstOptimizer(prefix,False);n.sync();result["engine_seconds"]=time.perf_counter()-t
      ids=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda");ids[0,0]=128000
      for length in [ARGS.length]:
        with torch.no_grad():h=prefix.forward_full(ids[:,:length])[0].float()
        refrow=next(v for v in old["runs"] if v["length"]==length and v["steps"]==64 and v["rep"]==0)
        if n.digest(ROOT/refrow["path"])!=refrow["sha256"]:raise RuntimeError("old public output changed")
        reference=n.load_file(str(ROOT/refrow["path"]));first=None
        for rep in range(3):
            output,stats=engine.decode(h,64)
            if first is None:first,first_stats=output,stats
            exact=inherited.equal(output,stats,reference,refrow["stats"]) and inherited.equal(output,stats,first,first_stats)
            entry={"length":length,"rep":rep,"stats":stats,"anchor_exact":exact,**save(f"control_{length}_{rep}",output)}
            result["runs"].append(entry)
            if not exact or not stats["numeric_valid"]:raise RuntimeError("unchanged anchor/repetition failed")
            guard()
        engine.decode(h,16);engine.stage="mirror"
        result["profiles"].append({"length":length,**profile(f"full_{length}",lambda:engine.train_step(length))})
        guard()
        engine.decode(h,16);engine.stage="mirror"
        engine.gradient.zero_();loss=engine.evaluate(length);loss.backward();del loss
        logits=engine.logits[:length-1].detach();gradient=engine.gradient[:length-1].detach();error=engine.position_error[:length-1].detach()
        fixture={"logits":logits.clone(),"gradient":gradient.clone(),"error":error.clone(),"public_ids":ids[:,:length],"observed":h}
        expected,expected_info=update(logits,gradient,error,2.,4)
        fixture["expected_logits"]=expected.clone()
        fixture.update({"info_"+k:v.clone() for k,v in expected_info.items()})
        result["update_fixtures"].append({"length":length,**save(f"update_fixture_{length}",fixture)})
        repeat_ok=[]
        def budget_call():
            got,info=update(logits,gradient,error,2.,4)
            # Equality checks stay outside the profiled CUDA work.
            return got,info
        result["profiles"].append({"length":length,**profile(f"budget_{length}",budget_call)})
        for _ in range(3):
            got,info=budget_call()
            repeat_ok.append(torch.equal(got,expected) and all(torch.equal(info[k],v) for k,v in expected_info.items()))
        unchanged=torch.equal(logits,fixture["logits"]) and torch.equal(gradient,fixture["gradient"]) and torch.equal(error,fixture["error"])
        result["update_fixtures"][-1].update(repeats_exact=all(repeat_ok),inputs_unchanged=unchanged)
        if not all(repeat_ok) or not unchanged:raise RuntimeError("isolated update changed")
        guard()
        if length==128:
            result["largest_cell"]={"passed":True,"peak_reserved":torch.cuda.max_memory_reserved()}
            print("LARGEST_CURRENT_OPTIMIZER_PROFILE_PASSED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        print("CURRENT_OPTIMIZER_PROFILE_COMPLETE",length,flush=True)
        del fixture,expected,expected_info,got,info
      if inherited.binding()!=bound:raise RuntimeError("inherited source changed")
      for p,sha in result["sources"].items():
          if n.digest(ROOT/p)!=sha:raise RuntimeError("source changed")
      result["passed"]=len(result["runs"])==3 and len(result["profiles"])==2
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
        peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
      n.write(DEST,result);print("CURRENT_OPTIMIZER_PROFILE_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

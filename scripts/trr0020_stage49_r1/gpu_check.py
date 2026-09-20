"""Qualify moment-bound updates on tiny and frozen full-vocabulary public states."""
from pathlib import Path
import sys,os,json,time,subprocess,resource,traceback
ROOT=Path(__file__).resolve().parents[2]
for part in ["scripts/trr0014","scripts/trr0020_stage32","scripts/trr0020_stage39","scripts/trr0020_stage44","scripts/trr0020_stage46"]:
    sys.path.insert(0,str(ROOT/part))
sys.path.insert(0,str(Path(__file__).parent))
import native as n
from gpu_qualification import guard
from budget_step import update as original
from reused_step import update as reused
from moment_step import update,MODES
from pointwise import bits_equal,TorchPointwise
from moment_cpu_reference import fixture,KINDS
torch=n.torch;X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev49_r1_gpu";DEST=X/"dev49_r1_gpu_check.json"
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism")
    cpu=json.loads((X/"dev49_r1_cpu_reference.json").read_text())
    profile=json.loads((X/"dev41_profile_r2.json").read_text())
    fresh=[json.loads((X/f"dev44_fixture_{length}.json").read_text()) for length in [128,40]]
    if not cpu["passed"] or not profile["passed"] or not all(v["passed"] for v in fresh):raise RuntimeError("unqualified inputs")
    paths=list(Path(__file__).parent.glob("*.py"))
    paths +=[ROOT/name for name in ["scripts/trr0020_stage44/pointwise.py","scripts/trr0020_stage46/reused_step.py","scripts/trr0020_stage32/budget_step.py","scripts/trr0014/native.py","scripts/trr0020_resource_guard.py"]]
    paths +=[X/name for name in ["DEV49_R1_PLAN.md","dev49_r1_preflight.json","dev49_r1_cpu_reference.json","dev41_profile_r2.json","dev44_fixture_128.json","dev44_fixture_40.json"]]
    result={"task_id":"TRR-0020","scope":"public update qualification; not decoder or benchmark accuracy",
      "passed":False,"truth_read":False,"shortlist":None,"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "cpu_sources":cpu["sources"],"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":[sys.executable,*sys.argv],"start_unix":time.time(),"cells":[],"benchmarks":[],"controls":[]}
    def check_sources():
        for path,sha in (result["sources"]|result["cpu_sources"]).items():
            if n.digest(ROOT/path)!=sha:raise RuntimeError("source changed:"+path)
    def save(label,values):
        path=OUT/(label+".safetensors");start=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in values.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-start}
    def equal(a,ai,b,bi):
        return bits_equal(a,b) and set(ai)==set(bi) and all(bits_equal(v,bi[k]) for k,v in ai.items())
    try:
      check_sources();guard()
      public=[]
      for length in [128,40]:
        q=next(v for v in fresh if v["length"]==length)
        for steps in [0,16,64,128]:
            source=(next(v for v in profile["update_fixtures"] if v["length"]==length) if steps==16 else next(v for v in q["fixtures"] if v["steps"]==steps))
            public.append((f"public_{length}_{steps}",source))
      cases=public+[(f"tiny_{v}_{kind}",(v,kind)) for v in [17,37,1031] for kind in KINDS]
      with torch.no_grad():
       for label,source in cases:
        guard();is_public=isinstance(source,dict)
        if is_public:
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("fixture changed")
            data=n.load_file(str(ROOT/source["path"]))
            z,g,e=[data[k].to("cuda") for k in ["logits","gradient","error"]]
        else:z,g,e=[v.to("cuda") for v in fixture(*source)];data=None
        initial=[v.clone() for v in [z,g,e]];p=z.softmax(-1);saved_p=p.clone()
        control,ci=original(z,g,e,2.,4)
        if is_public and (not bits_equal(control.cpu(),data["expected_logits"]) or not all(bits_equal(v.cpu(),data["info_"+k]) for k,v in ci.items())):
            raise RuntimeError("original public anchor changed")
        same,si=reused(z,g,e,2.,4,probability=p)
        if not equal(control,ci,same,si):raise RuntimeError("reused control differs")
        result["controls"].append({"label":label,"public_anchor":is_public,"reused_exact":True})
        expected={"control":(control.clone(),{k:v.clone() for k,v in ci.items()})}
        for mode in MODES:
            ref,ri=update(z,g,e,mode,probability=p,gpu_root=False,pw=TorchPointwise)
            cell={"label":label,"mode":mode,"rows":len(z),"vocab":z.shape[1],"fixture":source if is_public else None,"repetitions":[]}
            result["cells"].append(cell)
            for rep in range(3):
                got,info=update(z,g,e,mode,probability=p)
                active=info["active"]
                relative=((info["normalized_step"].double()-ri["normalized_step"].double()).abs()/ri["normalized_step"].double().abs().clamp_min(1e-30))
                tv=(got.double().softmax(-1)-ref.double().softmax(-1)).abs().sum(-1)*.5
                lp=z.double().log_softmax(-1);lq=got.double().log_softmax(-1)
                direct=(lp.exp()*(lp-lq)).sum(-1)
                gain=((lp.exp()-lq.exp())*g.double()).sum(-1)
                del lp,lq
                finite=all(bool(torch.isfinite(v).all()) for v in [got,*info.values(),direct,gain])
                feasible=bool((direct>=-2e-5).all() and (direct<=info["requested_budget"].double()+2e-5).all())
                if rep==0:expected[mode]=(got.clone(),{k:v.clone() for k,v in info.items()})
                repeated=equal(got,info,*expected[mode])
                row={"rep":rep,"finite":finite,"feasible":feasible,"exact_repeat":repeated,
                  "maximum_relative_root_error":float(relative.max()),"maximum_probability_tv":float(tv.max()),
                  "maximum_direct_minus_bound":float((direct-info["bound_kl"]).max()),
                  "mean_direct_kl":float(direct.mean()),"mean_linear_gain":float(gain.mean()),
                  "mean_budget_use":float((direct/info["requested_budget"].double().clamp_min(1e-30))[active].mean()) if bool(active.any()) else 0.,
                  **save(label+"_"+mode+f"_{rep}",{"logits":got,**{"info_"+k:v for k,v in info.items()},"direct_kl":direct,"linear_gain":gain})}
                cell["repetitions"].append(row)
                if not finite or not feasible or not repeated or row["maximum_relative_root_error"]>3e-6 or row["maximum_probability_tv"]>2e-5:
                    raise RuntimeError("GPU bound qualification failed")
                guard()
            del ref,ri,got,info,direct,gain,tv,relative
        if not all(bits_equal(a,b) for a,b in zip(initial,[z,g,e])) or not bits_equal(p,saved_p):raise RuntimeError("inputs mutated")
        if is_public:
            graphs={};setup={}
            def fn(name):
                return reused(z,g,e,2.,4,probability=p) if name=="control" else update(z,g,e,name,probability=p)
            for name in ["control",*MODES]:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());n.sync();start=time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(3):fn(name)
                torch.cuda.current_stream().wait_stream(stream);n.sync();warm=time.perf_counter()-start
                graph=torch.cuda.CUDAGraph();start=time.perf_counter()
                with torch.cuda.graph(graph,stream=stream):out,details=fn(name)
                n.sync();graphs[name]=(graph,out,details,stream);setup[name]={"warm_seconds":warm,"capture_seconds":time.perf_counter()-start}
            bench={"label":label,"rows":len(z),"vocab":z.shape[1],"setup":setup,"groups":[]}
            names=["control",*MODES]
            for group in range(3):
                for name in names[group:]+names[:group]:
                    graph,out,details,stream=graphs[name]
                    start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                    n.sync();start=time.perf_counter();start_event.record()
                    for _ in range(20):graph.replay()
                    end_event.record();n.sync()
                    wall=(time.perf_counter()-start)/20;gpu=start_event.elapsed_time(end_event)/20
                    if not equal(out,details,*expected[name]):raise RuntimeError("graph changed output")
                    bench["groups"].append({"group":group,"method":name,"calls":20,"seconds_per_call":wall,"gpu_ms_per_call":gpu,"eager_graph_exact":True})
                    guard()
            result["benchmarks"].append(bench)
            del graphs,graph,out,details,stream
        if label=="public_128_0":
            result["largest_cell"]={"passed":True,"peak_reserved":torch.cuda.max_memory_reserved()}
            print("LARGEST_MOMENT_BOUND_PASSED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        print("MOMENT_BOUND_CELL_COMPLETE",label,flush=True)
        del z,g,e,data,initial,p,saved_p,control,ci,same,si,expected
      if len(result["cells"])!=58 or len(result["benchmarks"])!=8:raise RuntimeError("incomplete qualification")
      check_sources();result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
      n.write(DEST,result);print("MOMENT_BOUND_QUALIFICATION_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

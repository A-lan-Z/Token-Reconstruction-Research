"""Require exact bytes before accepting an optimizer execution optimization."""
from pathlib import Path
import sys,os,json,time,subprocess,resource,traceback,statistics
ROOT=Path(__file__).resolve().parents[2]
for part in ["scripts/trr0014","scripts/trr0020_stage32","scripts/trr0020_stage39"]:sys.path.insert(0,str(ROOT/part))
sys.path.insert(0,str(Path(__file__).parent))
import native as n
from gpu_qualification import guard
from budget_step import update as original
from exact_step import update
from pointwise import TorchPointwise,TritonPointwise,bits_equal
from cpu_check import fixture
torch=n.torch;X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev44_gpu";DEST=X/"dev44_gpu_check.json"
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism")
    cpu=json.loads((X/"dev44_cpu_reference.json").read_text());profile=json.loads((X/"dev41_profile_r2.json").read_text())
    fresh=[json.loads((X/f"dev44_fixture_{v}.json").read_text()) for v in [128,40]]
    if not cpu["passed"] or not profile["passed"] or not all(v["passed"] for v in fresh):raise RuntimeError("unqualified inputs")
    paths=list(Path(__file__).parent.glob("*.py"))+[X/v for v in ["DEV44_PLAN.md","DEV44_GPU_PLAN.md","dev44_gpu_preflight.json","dev44_cpu_reference.json","dev41_profile_r2.json","dev44_fixture_128.json","dev44_fixture_40.json"]]
    result={"task_id":"TRR-0020","scope":"exact public full-vocabulary update qualification; no new decoder scores",
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "cpu_sources":cpu["sources"],"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":[sys.executable,*sys.argv],"start_unix":time.time(),"passed":False,
      "truth_read":False,"shortlist":None,"cells":[],"benchmarks":[],"primitive_checks":[]}
    def save(label,values):
        p=OUT/(label+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in values.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    def check_sources():
        for p,sha in (result["cpu_sources"]|result["sources"]).items():
            if n.digest(ROOT/p)!=sha:raise RuntimeError("source changed:"+p)
    current=[""];counts={}
    class Checking:
      def __getattr__(self,name):
        def fn(*args):
            expected=getattr(TorchPointwise,name)(*args);got=getattr(TritonPointwise,name)(*args)
            aa=expected if isinstance(expected,tuple) else (expected,);bb=got if isinstance(got,tuple) else (got,)
            exact=all(bits_equal(a,b) for a,b in zip(aa,bb))
            key=current[0]+"_"+name;index=counts.get(key,0);counts[key]=index+1
            entry={"case":current[0],"primitive":name,"index":index,"all_bytes_equal":exact}
            result["primitive_checks"].append(entry)
            if not exact:
                values={**{"expected_"+str(i):v for i,v in enumerate(aa)},**{"actual_"+str(i):v for i,v in enumerate(bb)}}
                values.update({"input_"+str(i):v for i,v in enumerate(args) if isinstance(v,torch.Tensor)})
                entry["failure_arrays"]=save("failed_"+key+"_"+str(index),values)
                entry["maximum_errors"]=[float((a-b).abs().max()) if a.dtype!=torch.bool else None for a,b in zip(aa,bb)]
                raise RuntimeError("primitive changed bytes:"+key)
            return got
        return fn
    try:
      check_sources();guard()
      public=[("profile_"+str(v["length"])+"_16",v) for v in sorted(profile["update_fixtures"],key=lambda v:-v["length"])]
      public += [(f"fresh_{q['length']}_{v['steps']}",v) for q in fresh for v in q["fixtures"]]
      cases=public+[(f"tiny_{v}_{kind}",(v,kind)) for v in [17,1023,1031] for kind in ["random","peaked","flat","zero_budget"]]
      with torch.no_grad():
       for label,source in cases:
        current[0]=label;is_public=isinstance(source,dict);guard()
        if is_public:
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("fixture changed")
            data=n.load_file(str(ROOT/source["path"]))
            logits=data["logits"].to("cuda");gradient=data["gradient"].to("cuda");error=data["error"].to("cuda")
        else:logits,gradient,error=[v.to("cuda") for v in fixture(*source)];data=None
        original_inputs=[v.clone() for v in [logits,gradient,error]]
        expected,info=original(logits,gradient,error,2.,4)
        anchor=(not is_public or (bits_equal(expected.cpu(),data["expected_logits"]) and all(bits_equal(v.cpu(),data["info_"+k]) for k,v in info.items())))
        if not anchor:raise RuntimeError("original full fixture changed")
        refactored,refinfo=update(logits,gradient,error,2.,4,pw=TorchPointwise)
        if not bits_equal(refactored,expected) or not all(bits_equal(refinfo[k],v) for k,v in info.items()):raise RuntimeError("torch refactor changed bytes")
        cell={"label":label,"rows":len(logits),"vocab":logits.shape[1],"public_fixture":source if is_public else None,"anchor_exact":anchor,"repetitions":[]}
        result["cells"].append(cell)
        checked,checkinfo=update(logits,gradient,error,2.,4,pw=Checking())
        for rep in range(3):
            got,details=update(logits,gradient,error,2.,4)
            arrays={"logits":got,**{"info_"+k:v for k,v in details.items()}}
            exact=bits_equal(got,expected) and all(bits_equal(details[k],v) for k,v in info.items())
            finite=all(bool(torch.isfinite(v).all()) for v in arrays.values())
            row={"rep":rep,"all_bytes_equal":exact,"finite":finite,**save(label+f"_{rep}",arrays)}
            cell["repetitions"].append(row)
            if not exact or not finite:raise RuntimeError("complete update changed bytes; preserved")
            guard()
        if not all(bits_equal(a,b) for a,b in zip(original_inputs,[logits,gradient,error])):raise RuntimeError("input mutated")
        cell["inputs_unchanged"]=True
        if is_public:
            graphs={};setup={}
            for name,fn in [("original",original),("exact_fused",update)]:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());n.sync();t=time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(3):fn(logits,gradient,error,2.,4)
                torch.cuda.current_stream().wait_stream(stream);n.sync();warm=time.perf_counter()-t
                graph=torch.cuda.CUDAGraph();t=time.perf_counter()
                with torch.cuda.graph(graph,stream=stream):out,details=fn(logits,gradient,error,2.,4)
                n.sync();graphs[name]=(graph,out,details,stream);setup[name]={"warm_seconds":warm,"capture_seconds":time.perf_counter()-t}
            bench={"label":label,"rows":len(logits),"vocab":logits.shape[1],"setup":setup,"groups":[]}
            for group in range(3):
                for name in (["original","exact_fused"] if group%2==0 else ["exact_fused","original"]):
                    graph,out,details,stream=graphs[name]
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    n.sync();t=time.perf_counter();begin.record()
                    for _ in range(20):graph.replay()
                    end.record();n.sync();wall=(time.perf_counter()-t)/20;gpu=begin.elapsed_time(end)/20
                    exact=bits_equal(out,expected) and all(bits_equal(details[k],v) for k,v in info.items())
                    if not exact:raise RuntimeError("graph changed bytes")
                    bench["groups"].append({"group":group,"method":name,"calls":20,"seconds_per_call":wall,"gpu_ms_per_call":gpu,"all_bytes_equal":exact})
                    guard()
            result["benchmarks"].append(bench)
            del graphs,graph,out,details,stream
        if label=="profile_128_16":
            result["largest_cell"]={"passed":True,"peak_reserved":torch.cuda.max_memory_reserved()}
            print("LARGEST_EXACT_UPDATE_PASSED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        print("EXACT_UPDATE_CELL_COMPLETE",label,flush=True)
        del logits,gradient,error,data,original_inputs,expected,info,refactored,refinfo,checked,checkinfo,arrays,got
      current[0]="signed_zero_subnormal"
      tiny=torch.finfo(torch.float32).tiny
      vals=torch.tensor([0.,-0.,tiny/8,-tiny/8,tiny,-tiny,1.,-1.],device="cuda").repeat(3,3)
      low=vals.amin(-1,keepdim=True);den=(vals.amax(-1,keepdim=True)-low).clamp_min(tiny)
      p=torch.full_like(vals,1/vals.shape[1]);check=Checking()
      d,pd=check.normalized(vals,low,den,p)
      c,pc,pcc=check.centered(d,pd.sum(-1,keepdim=True),p)
      tilted=check.tilt(vals,torch.tensor([0.,tiny,1.],device="cuda"),c)
      check.finish(tilted,tilted.amax(-1,keepdim=True),vals,vals.amax(-1,keepdim=True),torch.tensor([True,False,True],device="cuda"))
      if len(result["cells"])!=20 or len(result["benchmarks"])!=8:raise RuntimeError("incomplete")
      check_sources();result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
        peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
      n.write(DEST,result);print("EXACT_UPDATE_QUALIFICATION_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

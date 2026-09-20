"""All-vocabulary tiled scalar qualification and isolated replay timing."""
from pathlib import Path
import sys,os,json,time,subprocess,resource,traceback,statistics
ROOT=Path(__file__).resolve().parents[2]
for part in ["scripts/trr0014","scripts/trr0020_stage32","scripts/trr0020_stage39"]:sys.path.insert(0,str(ROOT/part))
import native as n
from gpu_qualification import guard
from budget_step import update as original
from fused_step import update
from tiled_evaluate import TiledEvaluator
from cpu_check import fixture
torch=n.torch;X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev42_gpu";DEST=X/"dev42_gpu_check.json"

def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism")
    cpu=json.loads((X/"dev42_cpu_reference.json").read_text());profile=json.loads((X/"dev41_profile_r2.json").read_text())
    if not cpu["passed"] or not profile["passed"]:raise RuntimeError("unqualified input")
    paths=list(Path(__file__).parent.glob("*.py"))+[X/v for v in ["DEV42_PLAN.md","dev42_gpu_preflight.json","dev42_cpu_reference.json","dev41_profile_r2.json"]]
    paths += [ROOT/"scripts/trr0014/native.py",ROOT/"scripts/trr0020_resource_guard.py",ROOT/"scripts/trr0020_stage39/gpu_qualification.py"]
    result={"task_id":"TRR-0020","scope":"public numerical kernel checks and isolated update speed; no decoder accuracy claim",
      "sources":cpu["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":[sys.executable,*sys.argv],"start_unix":time.time(),"passed":False,
      "truth_read":False,"shortlist":None,"cells":[],"benchmarks":[]}
    def save(label,values):
        p=OUT/(label+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in values.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    def sources():
        for p,sha in result["sources"].items():
            if n.digest(ROOT/p)!=sha:raise RuntimeError("source changed:"+p)
    try:
      sources();guard()
      cases=[("public_"+str(v["length"]),v) for v in sorted(profile["update_fixtures"],key=lambda v:-v["length"])]
      cases += [(f"tiny_{v}_{kind}",(v,kind)) for v in [17,1023,1031] for kind in ["random","peaked","flat","zero_budget"]]
      with torch.no_grad():
       for label,source in cases:
        guard();public=isinstance(source,dict)
        if public:
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("fixture changed")
            data=n.load_file(str(ROOT/source["path"]))
            logits=data["logits"].to("cuda");gradient=data["gradient"].to("cuda");error=data["error"].to("cuda")
        else:
            logits,gradient,error=[v.to("cuda") for v in fixture(*source,dtype=torch.float32)]
            data=None
        base_inputs=[v.clone() for v in [logits,gradient,error]]
        expected,old_info=original(logits,gradient,error,2.,4)
        if public and (not torch.equal(expected.cpu(),data["expected_logits"]) or not all(torch.equal(v.cpu(),data["info_"+k]) for k,v in old_info.items())):raise RuntimeError("original anchor changed")
        captured={};evaluations=[]
        class ReferenceEvaluator:
            def __init__(self,lp,c,b,corr):
                self.lp=lp;self.c=c;self.b=b;self.corr=corr
                captured.update(logp=lp,centered=c,base=b,correction=corr)
            def __call__(self,t):
                tilted=self.lp-t[:,None]*self.c;logz=torch.logsumexp(tilted,-1);p=torch.softmax(tilted,-1)
                value=logz-self.b+t*self.corr;deriv=self.corr-(p*self.c).sum(-1)
                evaluations.append((t.clone(),value.clone(),deriv.clone()))
                return value,deriv
        exact,exact_info=update(logits,gradient,error,2.,4,evaluator_factory=ReferenceEvaluator)
        if not torch.equal(exact,expected) or not all(torch.equal(exact_info[k],v) for k,v in old_info.items()):raise RuntimeError("refactoring control differs")
        evaluator=TiledEvaluator(captured["logp"],captured["centered"],captured["base"],captured["correction"])
        cell={"label":label,"rows":len(logits),"vocab":logits.shape[-1],"public_fixture":source if public else None,
          "original_anchor_exact":True,"refactoring_exact":True,"scalar_checks":[],"repetitions":[]}
        result["cells"].append(cell)
        index=torch.tensor(sorted(set([0,len(logits)//2,len(logits)-1])),device="cuda")
        lp=captured["logp"].index_select(0,index).cpu().double();c=captured["centered"].index_select(0,index).cpu().double()
        b=captured["base"].index_select(0,index).cpu().double();corr=captured["correction"].index_select(0,index).cpu().double()
        for j,(rate,old,old_derivative) in enumerate(evaluations):
            n.sync();t=time.perf_counter();value,derivative=evaluator(rate);n.sync();seconds=time.perf_counter()-t
            torch.testing.assert_close(value,old,rtol=5e-4,atol=5e-5)
            torch.testing.assert_close(derivative,old_derivative,rtol=5e-4,atol=5e-5)
            tt=rate.index_select(0,index).cpu().double();tilted=lp-tt[:,None]*c
            ref=tilted.logsumexp(-1)-b+tt*corr;refd=corr-(tilted.softmax(-1)*c).sum(-1)
            got=value.index_select(0,index).cpu().double();gotd=derivative.index_select(0,index).cpu().double()
            torch.testing.assert_close(got,ref,rtol=5e-4,atol=5e-5);torch.testing.assert_close(gotd,refd,rtol=5e-4,atol=5e-5)
            cell["scalar_checks"].append({"evaluation":j,"seconds_including_possible_compile":seconds,
              "maximum_value_error":float((value-old).abs().max()),"maximum_derivative_error":float((derivative-old_derivative).abs().max()),
              "CPU64rows":index.cpu().tolist(),"CPU64value":ref.tolist(),"CPU64derivative":refd.tolist(),
              "GPUvalue":got.tolist(),"GPUderivative":gotd.tolist(),"passed":True})
        first=None
        for rep in range(3):
            got,info=update(logits,gradient,error,2.,4)
            arrays={"logits":got,**{"info_"+k:v for k,v in info.items()}}
            if first is None:first={k:v.clone() for k,v in arrays.items()}
            exact_repeat=all(torch.equal(v,first[k]) for k,v in arrays.items())
            tv=((got.softmax(-1)-expected.softmax(-1)).abs().sum(-1)*.5).max()
            pp=logits.cpu().double().log_softmax(-1);qq=got.cpu().double().log_softmax(-1)
            kl=(pp.exp()*(pp-qq)).sum(-1);budget=(2*error).clamp(0,1).cpu().double()
            valid=(exact_repeat and all(bool(torch.isfinite(v).all()) for v in arrays.values()) and
              torch.equal(info["active"],old_info["active"]) and torch.equal(info["requested_budget"],old_info["requested_budget"]) and
              float(tv)<=5e-4 and bool((kl<=budget+2e-5).all()) and bool((kl>=-2e-5).all()))
            record={"rep":rep,"exact_repeat":exact_repeat,"probability_TV":float(tv),
              "maximum_logit_error":float((got-expected).abs().max()),
              "maximum_step_error":float((info["normalized_step"]-old_info["normalized_step"]).abs().max()),
              "max_KL_minus_budget":float((kl-budget).max()),"minimum_KL":float(kl.min()),"passed":valid,
              **save(label+f"_{rep}",arrays)}
            cell["repetitions"].append(record)
            if not valid:raise RuntimeError("full update qualification failed; preserved")
            guard()
        if not all(torch.equal(a,b) for a,b in zip([logits,gradient,error],base_inputs)):raise RuntimeError("input mutated")
        cell["inputs_unchanged"]=True
        if public:
            graphs={};setup={}
            for name,fn in [("original",original),("fused",update)]:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());n.sync();t=time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(3):fn(logits,gradient,error,2.,4)
                torch.cuda.current_stream().wait_stream(stream);n.sync();warm=time.perf_counter()-t
                graph=torch.cuda.CUDAGraph();t=time.perf_counter()
                with torch.cuda.graph(graph,stream=stream):out,inf=fn(logits,gradient,error,2.,4)
                n.sync();graphs[name]=(graph,out,inf,stream);setup[name]={"warm_seconds":warm,"capture_seconds":time.perf_counter()-t}
            bench={"label":label,"rows":len(logits),"vocab":logits.shape[1],"setup":setup,"groups":[]}
            for group in range(3):
                for name in (["original","fused"] if group%2==0 else ["fused","original"]):
                    graph,out,inf,stream=graphs[name];begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    n.sync();t=time.perf_counter();begin.record()
                    for _ in range(20):graph.replay()
                    end.record();n.sync();wall=(time.perf_counter()-t)/20;gpu=begin.elapsed_time(end)/20
                    ref=expected if name=="original" else first["logits"]
                    if not torch.equal(out,ref):raise RuntimeError("eager/replay mismatch")
                    bench["groups"].append({"group":group,"method":name,"calls":20,"seconds_per_call":wall,"gpu_ms_per_call":gpu,"eager_equal":True})
                    guard()
            result["benchmarks"].append(bench);del graphs
        if label=="public_128":
            result["largest_cell"]={"passed":True,"peak_reserved":torch.cuda.max_memory_reserved()}
            print("LARGEST_TILED_KL_PASSED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
        print("TILED_KL_CELL_COMPLETE",label,flush=True)
        del captured,evaluator,evaluations,arrays,first,logits,gradient,error,data,base_inputs,expected,got,info,old_info,exact,exact_info
      if len(result["cells"])!=14 or len(result["benchmarks"])!=2:raise RuntimeError("incomplete")
      sources();result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
        peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
      n.write(DEST,result);print("TILED_KL_QUALIFICATION_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

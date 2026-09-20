"""Freeze unchanged public cold and late update states in isolated geometries."""
from pathlib import Path
import sys,os,json,time,subprocess,resource,argparse,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage38_grid"))
import support as inherited
from budget_step import update as original
n=inherited.n;torch=n.torch;guard=inherited.guard
parser=argparse.ArgumentParser();parser.add_argument("--length",type=int,choices=[128,40],required=True);args=parser.parse_args()
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020"/f"dev44_fixture_{args.length}";DEST=X/f"dev44_fixture_{args.length}.json"
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cpu=json.loads((X/"dev44_cpu_reference.json").read_text())
    paths=list(Path(__file__).parent.glob("*.py"))+[X/v for v in ["DEV44_PLAN.md","DEV44_GPU_PLAN.md","dev44_gpu_preflight.json","dev44_cpu_reference.json"]]
    result={"task_id":"TRR-0020","length":args.length,"scope":"unchanged original public optimizer states; no benchmark labels",
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"inherited_binding":inherited.binding(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"environment":n.environment(),"start_unix":time.time(),"passed":False,"fixtures":[]}
    def save(label,values):
        p=OUT/(label+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in values.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
      guard()
      if torch.cuda.mem_get_info()[0]<10*2**30:raise RuntimeError("initial10GiBfree margin required")
      phase=json.loads((X/"dev43_phase_control.json").read_text())
      if n.digest(ROOT/phase["qualification_path"])!=phase["qualification_sha256"]:raise RuntimeError("reference changed")
      reference=json.loads((ROOT/phase["qualification_path"]).read_text())
      n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();result["prefix_seconds"]=time.perf_counter()-t
      t=time.perf_counter();engine=inherited.FirstOptimizer(prefix,False);n.sync();result["engine_seconds"]=time.perf_counter()-t
      ids=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda");ids[0,0]=128000
      with torch.no_grad():h=prefix.forward_full(ids[:,:args.length])[0].float()
      for steps in [128,64,0]:
        output,stats=engine.decode(h,steps)
        rr=next(v for v in reference["runs"] if v["length"]==args.length and v["steps"]==(steps or 128) and v["rep"]==0)
        if n.digest(ROOT/rr["path"])!=rr["sha256"]:raise RuntimeError("public output changed")
        ref=n.load_file(str(ROOT/rr["path"]))
        if steps:
            anchor=inherited.equal(output,stats,ref,rr["stats"])
        else:
            anchor=all(torch.equal(v,ref[k]) for k,v in output.items() if k.startswith("warm_") or k in ["initial_embedding","step0"])
            anchor=anchor and all(stats["warm"][k]==rr["stats"]["warm"][k] for k in inherited.old.TRACE_KEYS)
        if not anchor:raise RuntimeError("original public anchor")
        engine.stage="mirror";engine.gradient.zero_();loss=engine.evaluate(args.length);loss.backward();del loss
        with torch.no_grad():
            logits=engine.logits[:args.length-1].detach();gradient=engine.gradient[:args.length-1].detach();error=engine.position_error[:args.length-1].detach()
            expected,info=original(logits,gradient,error,2.,4)
            values={"logits":logits,"gradient":gradient,"error":error,"expected_logits":expected,"public_ids":ids[:,:args.length],"observed":h,
              **{"info_"+k:v for k,v in info.items()}}
            row={"length":args.length,"steps":steps,"anchor_exact":anchor,"decode_stats":stats,
              "anchor":{"path":rr["path"],"sha256":rr["sha256"]},**save(f"state_{steps}",values)}
            result["fixtures"].append(row)
            del logits,gradient,error,expected,info,values
        engine.stage="warm";guard()
        print("ORIGINAL_PUBLIC_STATE_FROZEN",args.length,steps,flush=True)
      for p,sha in result["sources"].items():
        if n.digest(ROOT/p)!=sha:raise RuntimeError("source changed")
      if inherited.binding()!=result["inherited_binding"]:raise RuntimeError("inherited binding changed")
      result["passed"]=len(result["fixtures"])==3
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
        peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
      n.write(DEST,result);print("ORIGINAL_PUBLIC_FIXTURES_DONE",args.length,result["passed"],flush=True)
if __name__=="__main__":main()

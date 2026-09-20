from moment_canonical_support import *
import os

def equivalent(out,stats,old,prior):return dev.equal(out,stats,old,prior)
def main():
    if (X/"validation.json").exists() or (X/"validation_outputs").exists():raise RuntimeError("create-only validation")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong environment")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bound=binding();env=n.environment();start=time.time();n.guard()
    prefix=n.load_prefix();engine=dev.MomentOptimizer(prefix,"bennett");checks=[]
    folder=X/"validation_outputs";folder.mkdir(parents=True,exist_ok=True)
    original=json.loads((ROOT/"experiments/TRR-0020/dev50_phase_bennett.json").read_text())
    if n.digest(ROOT/original["qualification_path"])!=original["qualification_sha256"]:raise RuntimeError("changed original qualification")
    original_qualification=json.loads((ROOT/original["qualification_path"]).read_text())
    def save(label,out,stats,equivalence,old_path=None,old_sha=None):
        p=folder/(label+".safetensors");n.save_file(out,str(p))
        free,total=torch.cuda.mem_get_info()
        row={"id":label,"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"stats":stats,
          "equal_to_original":equivalence,"original_path":old_path,"original_sha256":old_sha,
          "free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated()}
        n.write(p.with_suffix(".json"),row);checks.append(row)
        if equivalence is False:raise RuntimeError("execution equivalence failed; saved")
        if free<2*2**30 or torch.cuda.max_memory_reserved()>8*2**30:raise RuntimeError("geometry memory qualification failed; saved")
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200051)).to("cuda");fixture[0,0]=128000
    for length in [128,40]:
        with torch.no_grad():h=prefix.forward_full(fixture[:,:length])[0].float()
        prior=next(v for v in original_qualification["runs"] if v["length"]==length and v["steps"]==64 and v["rep"]==0)
        p=ROOT/prior["path"]
        if n.digest(p)!=prior["sha256"]:raise RuntimeError("changed public original")
        old=n.load_file(str(p))
        for rep in range(2):
            out,stats=engine.decode(h,64);save(f"public_{length}_{rep}",out,stats,equivalent(out,stats,old,prior["stats"]),prior["path"],prior["sha256"])
    devinput=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1/observations.safetensors"
    observed=n.load_file(str(devinput))
    for e in original["entries"]:
        if e["steps"]!=64:continue
        p=ROOT/e["path"]
        if n.digest(p)!=e["sha256"]:raise RuntimeError("changed development original")
        out,stats=engine.decode(observed[e["id"]],64)
        save("development_"+e["id"],out,stats,equivalent(out,stats,n.load_file(str(p)),e["stats"]),e["path"],e["sha256"])
    lengths=sorted({r["positions"] for r in json.loads((X/"metadata.json").read_text())},reverse=True)
    for length in lengths:
        with torch.no_grad():h=prefix.forward_full(fixture[:,:length])[0].float()
        out,stats=engine.decode(h,64);save("geometry_"+str(length),out,stats,None)
        print("GEOMETRY",length,round(torch.cuda.max_memory_reserved()/2**30,4),flush=True)
    # Re-enter evicted native lengths to catch execution changes after geometry churn.
    for length in [128,40]:
        with torch.no_grad():h=prefix.forward_full(fixture[:,:length])[0].float()
        prior=next(v for v in original_qualification["runs"] if v["length"]==length and v["steps"]==64 and v["rep"]==0)
        old=n.load_file(str(ROOT/prior["path"]));out,stats=engine.decode(h,64)
        save("after_churn_"+str(length),out,stats,equivalent(out,stats,old,prior["stats"]),prior["path"],prior["sha256"])
    expected={f"public_{length}_{rep}" for length in [128,40] for rep in range(2)}
    expected|={"development_"+e["id"] for e in original["entries"] if e["steps"]==64}
    expected|={"geometry_"+str(length) for length in lengths}
    expected|={"after_churn_"+str(length) for length in [128,40]}
    if len(lengths)!=38 or len(checks)!=52 or {v["id"] for v in checks}!=expected:raise RuntimeError("incomplete native qualification")
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(X/"validation.json",{"binding":bound,"passed":True,"checks":checks,"environment":env,
      "source_phase_sha256":n.digest(ROOT/"experiments/TRR-0020/dev50_phase_bennett.json"),
      "public_equivalence_cells":6,"development_equivalence_cells":8,"geometry_cells":len(lengths),
      "peak_reserved":torch.cuda.max_memory_reserved(),"capture_events":engine.capture_events,"mirror_capture_events":engine.mirror_capture_events,
      "start_unix":start,"end_unix":time.time(),"truth_read":False})
    print("EQUIVALENT_AND_ALL_NATIVE_GEOMETRIES_QUALIFIED",flush=True)
if __name__=="__main__":main()

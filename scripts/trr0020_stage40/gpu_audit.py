"""Frozen public fixtures: separate reduced solve error from Krylov convergence."""
from pathlib import Path
import sys,os,json,time,subprocess,resource,traceback,statistics
ROOT=Path(__file__).resolve().parents[2]
for part in ["scripts/trr0014","scripts/trr0020_stage21","scripts/trr0020_stage39"]:sys.path.insert(0,str(ROOT/part))
import native as n
from linearized_prefix import public_parameters
from homotopy import forward,jvp,global_gmres
from gpu_qualification import guard
from krylov_audit import arnoldi,direction
torch=n.torch;X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev40_gpu"
DEST=X/"dev40_gpu_audit.json"
MODES=["normal_ridge","svd_ridge","svd_unregularized"]
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("deterministic flags")
    cpu=json.loads((X/"dev40_cpu_reference.json").read_text());old=json.loads((X/"dev39_gpu_qualification.json").read_text())
    if not cpu["passed"] or not old["passed"]:raise RuntimeError("unqualified input")
    paths=list(Path(__file__).parent.glob("*.py"))+[X/v for v in ["DEV40_PLAN.md","dev40_gpu_preflight.json","dev40_cpu_reference.json","dev39_gpu_qualification.json"]]
    result={"task_id":"TRR-0020","scope":"public coupled linear audit; no reconstruction or benchmark",
      "sources":old["sources"]|cpu["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":[sys.executable,*sys.argv],"start_unix":time.time(),
      "truth_read":False,"shortlist":None,"passed":False,"points":[]}
    def check_sources():
        for p,sha in result["sources"].items():
            if n.digest(ROOT/p)!=sha:raise RuntimeError("source changed: "+p)
    def save(label,values):
        path=OUT/(label+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in values.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-t}
    try:
      check_sources();guard()
      if n.digest(n.ASSETS/"backup/prefix.safetensors")!=old["prefix_sha256"]:raise RuntimeError("prefix changed")
      result["prefix_sha256"]=old["prefix_sha256"]
      n.sync();t=time.perf_counter();prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup")
      prefix.config._attn_implementation="eager";layers=public_parameters(prefix);n.sync()
      result["preparation_seconds"]=time.perf_counter()-t
      with torch.no_grad():
       for length in [128,40]:
        fixture=next(v for v in old["fixtures"] if v["length"]==length)
        if n.digest(ROOT/fixture["path"])!=fixture["sha256"]:raise RuntimeError("fixture changed")
        data={k:v.to("cuda") for k,v in n.load_file(str(ROOT/fixture["path"])).items()}
        pe=prefix.rotary_emb(data["embedding"][None],torch.arange(length,device="cuda")[None]);co,si=pe[0][0],pe[1][0]
        for kind in ["activation","embedding"]:
         for strength in [1.,.5]:
          label=f"{length}_{kind}_{strength}";lam=torch.tensor(strength,device="cuda");point=data[kind]
          output,cache=forward(point,layers,co,si,lam)
          b=-jvp(torch.zeros_like(point),layers,cache,co,si,lam,1.)[1:]
          def mv(v):return jvp(torch.cat([torch.zeros_like(v[:1]),v]),layers,cache,co,si,lam)[1:]
          prior=next(v for v in old["linear_cells"] if v["length"]==length and v["point"]==kind and v["strength"]==strength and v["steps"]==16)["repetitions"][0]
          if n.digest(ROOT/prior["path"])!=prior["sha256"]:raise RuntimeError("prior changed")
          anchor=n.load_file(str(ROOT/prior["path"]))
          if not torch.equal(b.cpu(),anchor["rhs"]):raise RuntimeError("RHS anchor changed")
          entry={"length":length,"point":kind,"strength":strength,"fixture":fixture,"prior":prior,"repetitions":[]}
          result["points"].append(entry);first=None
          for rep in range(3):
            n.sync();t=time.perf_counter();basis,h,beta=arnoldi(mv,b,32);n.sync();arnoldi_time=time.perf_counter()-t
            if rep==0:
                basis16,h16,beta16=arnoldi(mv,b,16)
                if not torch.equal(h[:17,:16],h16) or not torch.equal(basis[...,:17],basis16) or not torch.equal(beta,beta16):raise RuntimeError("Arnoldi prefix changed")
                original,stats=global_gmres(mv,b,16,1e-6)
                own,_,_=direction(basis,h,beta,16,"normal_ridge")
                if not torch.equal(own,original) or not torch.equal(own.cpu(),anchor["direction"]):raise RuntimeError("original answer changed")
                entry["original_exact"]=True;entry["prefix_exact"]=True
                del basis16,h16,original,own
            arrays={"basis":basis,"hessenberg":h,"rhs":b,"beta":beta.reshape(1)}
            comparisons=[]
            for steps in [16,32]:
              hh=h[:steps+1,:steps].cpu().double()
              singular=torch.linalg.svdvals(hh);rank=int((singular>singular.max()*1e-12).sum())
              for mode in MODES:
                n.sync();t=time.perf_counter();answer,coefficient,projected=direction(basis,h,beta,steps,mode)
                n.sync();solve_time=time.perf_counter()-t
                n.sync();t=time.perf_counter();actual=mv(answer)-b;n.sync();residual_time=time.perf_counter()-t
                key=f"{steps}_{mode}"
                arrays.update({key+"_direction":answer,key+"_coefficient":coefficient,key+"_projected":projected,key+"_residual":actual})
                comparisons.append({"steps":steps,"mode":mode,"solve_seconds":solve_time,"residual_check_seconds":residual_time,
                  "relative_residual":float(actual.norm()/beta),"projected_relative_residual":float(projected.norm()/beta),
                  "direction_norm":float(answer.norm()),"reduced_rank":rank,
                  "condition":float(singular.max()/singular.min().clamp_min(1e-30))})
            flat=basis[...,:32].reshape(-1,32);gram=flat.T@flat
            orth=float((gram-torch.eye(32,device="cuda")).abs().max())
            valid=all(bool(torch.isfinite(v).all()) for v in arrays.values())
            if first is None:first={k:v.clone() for k,v in arrays.items()}
            exact=all(torch.equal(v,first[k]) for k,v in arrays.items())
            record={"rep":rep,"arnoldi_seconds":arnoldi_time,"basis_orthogonality_maximum_error":orth,
              "comparisons":comparisons,"exact_repeat":exact,"finite":valid,**save(label+f"_{rep}",arrays)}
            entry["repetitions"].append(record)
            if not valid or not exact:raise RuntimeError("nonfinite or nonrepeatable arrays; preserved")
            guard()
          if length==128 and kind=="activation" and strength==1.:
            result["largest_cell"]={"passed":True,"steps":32,"peak_reserved":torch.cuda.max_memory_reserved()}
            print("LARGEST_KRYLOV_AUDIT_PASSED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
          print("KRYLOV_AUDIT_POINT_COMPLETE",length,kind,strength,flush=True)
          del cache,arrays,first,basis,h,b,flat,gram
        del data
      if len(result["points"])!=8 or sum(len(v["repetitions"]) for v in result["points"])!=24:raise RuntimeError("incomplete")
      check_sources();result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
      n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
        peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
      n.write(DEST,result);print("KRYLOV_AUDIT_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

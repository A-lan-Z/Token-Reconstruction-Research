"""Native-module JVP qualification and public coupled linear systems."""
from pathlib import Path
import sys,os,json,time,subprocess,resource,traceback,statistics
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"));sys.path.insert(0,str(ROOT/"scripts/trr0020_stage21"))
import native as n
from linearized_prefix import public_parameters,forward as previous_forward
from homotopy import forward,jvp,global_gmres
torch=n.torch;X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev39_gpu";DEST=X/"dev39_gpu_qualification.json"
STRENGTHS=[1.,.75,.5,.25,0.];BUDGETS=[16,8,4]
def guard():
    free,_=torch.cuda.mem_get_info()
    available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
    temp=int(subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],text=True))
    if free<3*2**30 or available<8*2**30 or torch.cuda.max_memory_reserved()>8*2**30 or temp>=80:raise RuntimeError("resource margin violated")
def native_scaled(prefix,value,strength,pe,mask):
    def scale_output(module,args,output):return output*strength
    handles=[]
    for layer in prefix.layers:
        handles.extend([layer.self_attn.o_proj.register_forward_hook(scale_output),layer.mlp.down_proj.register_forward_hook(scale_output)])
    try:
        hidden=value[None]
        for layer in prefix.layers:
            hidden=prefix._hidden(layer(hidden,position_embeddings=pe,attention_mask=mask,use_cache=False))
        return hidden[0]
    finally:
        for handle in handles:handle.remove()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("deterministic flags")
    cpu=json.loads((X/"dev39_cpu_reference.json").read_text())
    if not cpu["passed"]:raise RuntimeError("unqualified primitive")
    for path,sha in cpu["sources"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("CPU source changed")
    paths=list(Path(__file__).parent.glob("*.py"))+[X/r for r in ["DEV39_GPU_PLAN.md","dev39_gpu_preflight.json","dev39_cpu_reference.json"]]
    paths +=[ROOT/r for r in ["scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py","scripts/trr0020_resource_guard.py"]]
    result={"task_id":"TRR-0020","scope":"public actual-prefix homotopy derivatives and linear systems; no reconstruction",
      "sources":cpu["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "environment":n.environment(),"command":sys.argv,"start_unix":time.time(),"truth_read":False,"shortlist":None,
      "passed":False,"derivative_cells":[],"linear_cells":[],"fixtures":[]}
    def save(name,data):
        path=OUT/(name+".safetensors");t=time.perf_counter();n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard();result["prefix_sha256"]=n.digest(n.ASSETS/"backup/prefix.safetensors");result["config_sha256"]=n.digest(n.ASSETS/"backup/config.json")
        n.sync();t=time.perf_counter();prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);n.sync();result["preparation_seconds"]=time.perf_counter()-t
        for length in [128,40]:
            generator=torch.Generator().manual_seed(39039+length)
            ids=torch.randint(0,len(prefix.embed_tokens.weight),(1,length),generator=generator).to("cuda");ids[0,0]=128000
            with torch.no_grad():
                embedding=prefix.embed_tokens(ids)[0];observed=prefix.forward_full(ids)[0]
                activation=observed.clone();activation[:1].copy_(embedding[:1])
                tangent=torch.randn(length,embedding.shape[1],generator=generator).to("cuda")
                tangent=tangent*(embedding.norm(dim=-1).median()/embedding.shape[1]**.5);tangent[0].zero_()
                pe=prefix.rotary_emb(embedding[None],torch.arange(length,device="cuda")[None]);co,si=pe[0][0],pe[1][0]
                mask=prefix._causal_mask(embedding[None],start_pos=0,total_tokens=length)
            result["fixtures"].append({"length":length,"seed":39039+length,**save("fixture_"+str(length),{"public_ids":ids,"embedding":embedding,"activation":activation,"tangent":tangent})})
            for kind,point in [("activation",activation),("embedding",embedding)]:
              for strength in STRENGTHS:
                lam=torch.tensor(strength,device="cuda")
                with torch.no_grad():
                    output,cache=forward(point,layers,co,si,lam)
                    derivatives={"input":jvp(tangent,layers,cache,co,si,lam),
                      "strength":jvp(torch.zeros_like(point),layers,cache,co,si,lam,1.),
                      "mixed":jvp(tangent,layers,cache,co,si,lam,.37)}
                    again,repeat_cache=forward(point,layers,co,si,lam)
                    if not torch.equal(output,again):raise RuntimeError("forward repeatability")
                    for name,dx,dl in [("input",tangent,0.),("strength",torch.zeros_like(point),1.),("mixed",tangent,.37)]:
                        if not torch.equal(derivatives[name],jvp(dx,layers,repeat_cache,co,si,lam,dl)):raise RuntimeError("JVP repeatability")
                    del repeat_cache
                    if strength==0 and not torch.equal(output,point):raise RuntimeError("identity endpoint")
                    if strength==1 and not torch.equal(output,previous_forward(point,layers,co,si)[0]):raise RuntimeError("full endpoint")
                arrays={"input":point,"output":output};errors={}
                for name,dx,dl in [("input",tangent,0.),("strength",torch.zeros_like(point),1.),("mixed",tangent,.37)]:
                    with torch.enable_grad():
                        native_out,expected=torch.autograd.functional.jvp(lambda a,b:native_scaled(prefix,a,b,pe,mask),(point,lam),(dx,torch.tensor(dl,device="cuda")))
                    arrays[name+"_jvp"]=derivatives[name];arrays[name+"_reference"]=expected;arrays["native_output"]=native_out
                    torch.testing.assert_close(output,native_out,rtol=5e-4,atol=5e-5)
                    torch.testing.assert_close(derivatives[name],expected,rtol=5e-4,atol=5e-5)
                    errors[name+"_maximum_error"]=float((derivatives[name]-expected).abs().max())
                    errors[name+"_relative_norm_error"]=float((derivatives[name]-expected).norm()/expected.norm().clamp_min(1e-12))
                if not torch.equal(derivatives["input"][0],torch.zeros_like(derivatives["input"][0])):raise RuntimeError("BOS input tangent changed")
                label=f"{length}_{kind}_{strength}"
                result["derivative_cells"].append({"length":length,"point":kind,"strength":strength,
                  "forward_maximum_error":float((output-native_out).abs().max()),**errors,"passed":True,**save("derivative_"+label,arrays)})
                with torch.no_grad():
                    b=-derivatives["strength"][1:]
                    def matvec(value):
                        return jvp(torch.cat([torch.zeros_like(value[:1]),value],0),layers,cache,co,si,lam)[1:]
                    for steps in BUDGETS:
                        row={"length":length,"point":kind,"strength":strength,"steps":steps,"repetitions":[]};result["linear_cells"].append(row)
                        first=None
                        for rep in range(3):
                            n.sync();t=time.perf_counter();direction,stats=global_gmres(matvec,b,steps,1e-6);n.sync();seconds=time.perf_counter()-t
                            values={"direction":direction,"rhs":b,"residual_norm":stats["residual_norm"].reshape(1),"rhs_norm":stats["rhs_norm"].reshape(1)}
                            valid=all(bool(torch.isfinite(v).all()) for v in values.values())
                            equal=first is None or all(torch.equal(v,first[k]) for k,v in values.items())
                            if first is None:first={k:v.clone() for k,v in values.items()}
                            entry={"rep":rep,"seconds":seconds,"matvec_calls":stats["matvec_calls"],"exact_repeat":equal,
                              "relative_residual":float(stats["residual_norm"]/stats["rhs_norm"].clamp_min(1e-12)),
                              "direction_norm":float(direction.norm()),"passed":valid,**save(f"linear_{label}_{steps}_{rep}",values)}
                            row["repetitions"].append(entry)
                            if not valid or not equal:raise RuntimeError("invalid linear solution; preserved")
                            guard()
                    if length==128 and kind=="activation" and strength==1.:
                        result["largest_cell"]={"passed":True,"length":128,"steps":16,"peak_reserved":torch.cuda.max_memory_reserved()}
                        print("LARGEST_HOMOTOPY_CELL_QUALIFIED",round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
                print("HOMOTOPY_CELL_COMPLETE",length,kind,strength,flush=True)
                del cache,arrays,derivatives,native_out,expected,output,again
        if len(result["derivative_cells"])!=20 or len(result["linear_cells"])!=60:raise RuntimeError("incomplete matrix")
        for path,sha in result["sources"].items():
            if n.digest(ROOT/path)!=sha:raise RuntimeError("source changed during run")
        result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("HOMOTOPY_PUBLIC_QUALIFICATION_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

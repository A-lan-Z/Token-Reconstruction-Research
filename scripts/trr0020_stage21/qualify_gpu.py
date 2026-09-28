"""Public-fixture qualification of FP32 forward and own-position derivatives."""
from pathlib import Path
import sys,os,json,time,traceback,subprocess,resource,hashlib
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
import torch
from linearized_prefix import forward,diagonal_jvp,krylov_least_squares,public_parameters
X=ROOT/"experiments/TRR-0020"
OUT=ROOT/"outputs/TRR-0020/dev21_qualification"
DEST=X/"dev21_gpu_qualification.json"
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only qualification already exists")
    torch.set_num_threads(2);torch.manual_seed(200049);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    OUT.mkdir(parents=True)
    started=time.time()
    sources=[Path(__file__),ROOT/"scripts/trr0020_stage21/linearized_prefix.py",ROOT/"scripts/trr0014/native.py",ROOT/"scripts/agent4/common.py",ROOT/"src/token_reconstruction/public_prefix.py",X/"DEV21_GPU_QUALIFICATION.md",X/"dev21_qualification_preflight.json"]
    receipt={"task_id":"TRR-0020","kind":"public_fixture_numerical_qualification","environment":n.environment(),
      "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":sys.argv,"start_unix":started,"sources":{str(p.relative_to(ROOT)):digest(p) for p in sources},
      "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),"config_sha256":n.digest(n.ASSETS/"backup/config.json"),
      "seed":200049,"truth_read":False,"shortlist":None,"raw":[],"checks":[],"passed":False,
      "tolerances":{"forward_rtol":5e-4,"forward_atol":2e-5,"jvp_rtol":5e-4,"jvp_atol":5e-5}}
    def save(name,data):
        p=OUT/(name+".safetensors")
        if p.exists():raise RuntimeError("duplicate raw output")
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        receipt["raw"].append({"path":str(p.relative_to(ROOT)),"sha256":digest(p)})
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup")
        prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);n.sync()
        receipt["prefix_and_parameter_preparation_seconds"]=time.perf_counter()-t
        receipt["parameter_bytes"]=sum(p.numel()*p.element_size() for p in prefix.parameters())
        ids=torch.randint(256,128000,(128,),generator=torch.Generator().manual_seed(200049)).to("cuda")
        ids[0]=128000
        for length in [128,40]:
            base=prefix.embed_tokens(ids[:length]).detach()
            noise=torch.randn(base.shape,device="cpu",generator=torch.Generator().manual_seed(200049+length)).to("cuda")*.003
            noise[0]=0;x=base+noise
            positions=torch.arange(length,device="cuda")[None]
            cos,sin=prefix.rotary_emb(x[None],positions)
            mask=prefix._causal_mask(x[None],start_pos=0,total_tokens=length)
            def reference(value):
                h=value[None]
                for layer in prefix.layers:
                    h=prefix._hidden(layer(h,attention_mask=mask,position_ids=positions,use_cache=False,position_embeddings=(cos,sin)))
                return h[0]
            n.sync();t=time.perf_counter()
            with torch.no_grad():
                actual,cache=forward(x,layers,cos[0],sin[0]);expected=reference(x)
                target=reference(base)
            n.sync();forward_seconds=time.perf_counter()-t
            save(f"{length}_forward",{"input":x,"actual":actual,"reference":expected,"public_target":target,"public_ids":ids[:length]})
            item={"length":length,"kind":"independent_HF_FP32_forward","max_absolute_error":float((actual-expected).abs().max()),"relative_l2_error":float((actual-expected).norm()/expected.norm()),"seconds_including_three_forwards":forward_seconds}
            receipt["checks"].append(item)
            torch.testing.assert_close(actual,expected,rtol=5e-4,atol=2e-5);item["passed"]=True
            for pos in [1,length//2,length-1]:
                direction=torch.randn((2048,),generator=torch.Generator().manual_seed(200049+length+pos)).to("cuda")/(2048**.5)
                tangent=torch.zeros_like(x);tangent[pos]=direction
                n.sync();t=time.perf_counter()
                with torch.no_grad():product=diagonal_jvp(tangent,layers,cache,cos[0],sin[0])[pos]
                _,all_reference=torch.autograd.functional.jvp(reference,x,tangent,create_graph=False,strict=True)
                expected_product=all_reference[pos].detach();n.sync();seconds=time.perf_counter()-t
                save(f"{length}_jvp_{pos}",{"tangent":direction,"actual":product,"reference":expected_product,"future_reference":all_reference[pos+1:]})
                check={"length":length,"position":pos,"kind":"independent_HF_autograd_direction","max_absolute_error":float((product-expected_product).abs().max()),"relative_l2_error":float((product-expected_product).norm()/expected_product.norm()),"seconds":seconds}
                receipt["checks"].append(check)
                torch.testing.assert_close(product,expected_product,rtol=5e-4,atol=5e-5);check["passed"]=True
                del all_reference,expected_product,product,tangent
                n.guard()
            with torch.no_grad():
                rhs=target-actual;rhs[0]=0
                n.sync();t=time.perf_counter()
                direction,stats=krylov_least_squares(lambda v:diagonal_jvp(v,layers,cache,cos[0],sin[0]),rhs,steps=8,ridge=1e-6)
                independent_residual=(diagonal_jvp(direction,layers,cache,cos[0],sin[0])-rhs).norm(dim=-1)
                n.sync();seconds=time.perf_counter()-t
                save(f"{length}_linear",{"rhs":rhs,"direction":direction,"reported_residual":stats["residual_norm"],"recomputed_residual":independent_residual})
                check={"length":length,"kind":"actual_prefix_linear_solve","matvec_calls":stats["matvec_calls"],"recompute_matvec_calls":1,
                  "seconds":seconds,"mean_relative_residual":float((independent_residual[1:]/rhs[1:].norm(dim=-1)).mean()),
                  "max_relative_residual":float((independent_residual[1:]/rhs[1:].norm(dim=-1)).max())}
                receipt["checks"].append(check)
                assert torch.equal(direction[0],torch.zeros_like(direction[0]))
                assert bool(torch.isfinite(direction).all())
                torch.testing.assert_close(stats["residual_norm"],independent_residual,rtol=0,atol=0);check["passed"]=True
            n.guard();del x,base,cache,actual,expected,target,direction,rhs
            print("QUALIFIED_PUBLIC_GEOMETRY",length,flush=True)
        free,total=torch.cuda.mem_get_info()
        if free<3*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError("qualification memory margin")
        receipt.update(passed=True,free_bytes=free)
    except Exception:
        receipt["failure"]=traceback.format_exc()
        raise
    finally:
        n.sync()
        receipt.update(end_unix=time.time(),wall_seconds=time.time()-started,peak_reserved=torch.cuda.max_memory_reserved(),
          peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,receipt)
        print("QUALIFICATION_RECEIPT",str(DEST),receipt["passed"],flush=True)
if __name__=="__main__":main()

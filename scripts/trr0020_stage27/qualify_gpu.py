"""Actual-prefix qualification of committed-context current-token inversion."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020_stage21","scripts/trr0020_stage23","scripts/trr0020_stage24"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
import torch
from torch.nn import functional as F
from linearized_prefix import public_parameters
from single_position import forward,jvp,vjp,commit
from normalized import equation
from stable_cgls import least_squares
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev27_qualification"
DEST=X/"dev27_gpu_qualification.json"

class FrozenPast:
    def __init__(self,past):self.past=past;self.updated=set()
    def update(self,key,value,layer_idx,cache_kwargs=None):
        if layer_idx in self.updated:raise RuntimeError("reference cache reused")
        self.updated.add(layer_idx);old=self.past[layer_idx]
        return torch.cat([old[0][None],key],dim=2),torch.cat([old[1][None],value],dim=2)

@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only qualification exists")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    cpu=json.loads((X/"dev27_cpu_reference_r1.json").read_text())
    if not cpu["result"]["passed"]:raise RuntimeError("CPU qualification failed")
    for path,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/path)!=sha:raise RuntimeError("qualified CPU source changed")
    paths=list((ROOT/"scripts/trr0020_stage27").glob("*.py"))
    paths +=[ROOT/p for p in ["scripts/trr0020_stage21/linearized_prefix.py","scripts/trr0020_stage23/adjoint.py",
      "scripts/trr0020_stage24/normalized.py","scripts/trr0014/native.py","scripts/agent4/common.py",
      "src/token_reconstruction/public_prefix.py","scripts/trr0020_resource_guard.py"]]
    paths +=[X/p for p in ["DEV27_QUALIFICATION.md","dev27_preflight.json","dev27_cpu_reference_r1.json","dev21_gpu_qualification.json"]]
    q={"task_id":"TRR-0020","kind":"public_committed_context_qualification","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"contexts":[],"forwards":[],"derivatives":[],"linear_checks":[],"passed":False,
      "scope":"public random-token numerical fixture with known synthetic past; not hidden-input reconstruction"}
    def save(name,data):
        path=OUT/(name+".safetensors");start=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"save_and_hash_seconds":time.perf_counter()-start}
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);radius=prefix.embed_tokens.weight.norm(dim=-1).median()
        n.sync();q["prefix_preparation_seconds"]=time.perf_counter()-t;q["direction_radius"]=float(radius)
        old=json.loads((X/"dev21_gpu_qualification.json").read_text())
        for length in [128,40]:
            source=next(v for v in old["raw"] if v["path"].endswith(f"/{length}_forward.safetensors"))
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("changed public fixture")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda")
            ids=fixture["public_ids"];base=prefix.embed_tokens(ids);target=fixture["public_target"];noisy=fixture["input"]
            positions=torch.arange(length,device="cuda")[None];pe=prefix.rotary_emb(base[None],positions);cos,sin=pe[0][0],pe[1][0]
            independent=prefix.forward_full(ids[None])[0]
            if not torch.equal(independent,target):raise RuntimeError("public target anchor changed")
            selected=[length-1,length//2,1];snapshots={};output=[]
            past=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in layers]
            n.sync();t=time.perf_counter()
            for pos in range(length):
                if pos in selected:snapshots[pos]=past
                value,_,current=forward(base[pos:pos+1],layers,past,cos[pos:pos+1],sin[pos:pos+1])
                output.append(value);past=commit(past,current)
            n.sync();commit_seconds=time.perf_counter()-t
            observed=torch.cat(output,dim=0)
            row={"length":length,"source":source,"commit_forwards":length,"commit_seconds":commit_seconds,
              "max_forward_error":float((observed-independent).abs().max()),
              **save(f"context_{length}",{"actual":observed,"independent":independent})}
            q["contexts"].append(row)
            torch.testing.assert_close(observed,independent,rtol=5e-4,atol=5e-5)
            for pos in selected:
                cur=noisy[pos:pos+1];history=snapshots[pos];co,si=cos[pos:pos+1],sin[pos:pos+1]
                raw,caches,_=forward(cur,layers,history,co,si)
                position=torch.tensor([[pos]],device="cuda");rotary=(co[None],si[None])
                def reference(value):
                    hidden=value[None];cache=FrozenPast(history)
                    for layer in prefix.layers:
                        hidden=prefix._hidden(layer(hidden,attention_mask=None,position_ids=position,
                          use_cache=True,cache_position=position[0],position_embeddings=rotary,past_key_values=cache))
                    if len(cache.updated)!=len(layers):raise RuntimeError("incomplete reference cache")
                    return hidden[0]
                exact=reference(cur)
                row={"length":length,"position":pos,"max_error":float((raw-exact).abs().max()),
                  **save(f"forward_{length}_{pos}",{"input":cur,"actual":raw,"reference":exact})}
                q["forwards"].append(row)
                torch.testing.assert_close(raw,exact,rtol=5e-4,atol=5e-5)
                raw_mv=lambda v:jvp(v,layers,caches,co,si)
                raw_rmv=lambda v:vjp(v,layers,caches,co,si)
                rhs,mv,rmv=equation(raw,target[pos:pos+1],raw_mv,raw_rmv)
                for mode in ["raw","normalized"]:
                    jfun,jtfun=(raw_mv,raw_rmv) if mode=="raw" else (mv,rmv)
                    ref=reference if mode=="raw" else lambda v:F.normalize(reference(v),dim=-1)
                    vector=torch.randn(cur.shape,generator=torch.Generator().manual_seed(200065+length+pos)).to("cuda")/(2048**.5)
                    a=jfun(vector);at=jtfun(vector)
                    with torch.enable_grad():
                        _,ej=torch.autograd.functional.jvp(ref,cur,vector,create_graph=False,strict=True)
                        variable=cur.detach().requires_grad_(True);y=ref(variable)
                        ejt=torch.autograd.grad(y,variable,grad_outputs=vector)[0].detach()
                    ej=ej.detach()
                    row={"length":length,"position":pos,"mode":mode,"jvp_max_error":float((a-ej).abs().max()),"vjp_max_error":float((at-ejt).abs().max()),
                      **save(f"derivatives_{length}_{pos}_{mode}",{"vector":vector,"jvp":a,"reference_jvp":ej,"vjp":at,"reference_vjp":ejt})}
                    q["derivatives"].append(row)
                    torch.testing.assert_close(a,ej,rtol=5e-4,atol=5e-5);torch.testing.assert_close(at,ejt,rtol=5e-4,atol=5e-5)
                    row["passed"]=True;del variable,y,ej,ejt,a,at
                    n.guard()
                for steps in [16,8,4]:
                    for ridge in [.0001,.01]:
                        reps=[];first=None
                        for rep in range(3):
                            n.sync();t=time.perf_counter();direction,stats=least_squares(mv,rmv,rhs,steps,ridge);n.sync();seconds=time.perf_counter()-t
                            residual=mv(direction)-rhs
                            factor=(radius/direction.norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
                            one,_,_=forward(cur+factor[:,None]*direction,layers,history,co,si)
                            half,_,_=forward(cur+.5*factor[:,None]*direction,layers,history,co,si)
                            target_unit=F.normalize(target[pos:pos+1],dim=-1)
                            one_error=F.normalize(one,dim=-1)-target_unit;half_error=F.normalize(half,dim=-1)-target_unit
                            data={"direction":direction,"recurrence_residual":stats["recurrence_residual_norm"],"linear_residual":residual,"one_step_residual":one_error,
                              "half_step_residual":half_error,"ridge":stats["ridge"],"clip_factor":factor,"frozen_iteration":stats["frozen_iteration"]}
                            artifact=save(f"linear_{length}_{pos}_{steps}_{ridge}_{rep}",data)
                            same=True if first is None else all(torch.equal(v,first[k]) for k,v in data.items())
                            if first is None:first={k:v.clone() for k,v in data.items()}
                            row={"rep":rep,"seconds":seconds,"repeat_equal":same,"linear_relative_residual":float(residual.norm()/rhs.norm()),
                              "one_step_relative_residual":float(one_error.norm()/rhs.norm()),"half_step_relative_residual":float(half_error.norm()/rhs.norm()),
                              "frozen_iteration":int(stats["frozen_iteration"][0]),**artifact};reps.append(row)
                            if not same or not all(torch.isfinite(v).all() for v in data.values()):raise RuntimeError("invalid/repeated direction; saved")
                            torch.testing.assert_close(stats["recurrence_residual_norm"],residual.norm(dim=-1),rtol=5e-4,atol=5e-6);n.guard()
                        q["linear_checks"].append({"length":length,"position":pos,"steps":steps,"ridge":ridge,"repetitions":reps,
                          "jvp_calls_per_solve":steps,"vjp_calls_per_solve":steps+1,"validation_jvp_calls_per_solve":1,"validation_forward_calls_per_solve":2})
                print("CURRENT_POSITION_QUALIFIED",length,pos,flush=True)
        if len(q["derivatives"])!=12 or len(q["linear_checks"])!=36:raise RuntimeError("incomplete qualification")
        q["passed"]=True
    except Exception:
        q["failure"]=traceback.format_exc();raise
    finally:
        n.sync();q.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,q);print("CURRENT_TOKEN_QUALIFICATION",q["passed"],flush=True)
if __name__=="__main__":main()

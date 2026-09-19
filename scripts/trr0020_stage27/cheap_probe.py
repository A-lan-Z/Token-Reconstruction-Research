"""Public test of current-token directions with one or two derivative products."""
from pathlib import Path
import json,sys,os,time,subprocess,traceback,resource
from qualify_gpu import ROOT,n,torch,F,public_parameters,forward,jvp,vjp,commit,equation,least_squares
from cheap_update import direction,RULES
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev27_cheap"
DEST=X/"dev27_cheap_probe.json"
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only cheap probe")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("deterministic environment")
    cpu=json.loads((X/"dev27_cheap_cpu_reference.json").read_text())
    old=json.loads((X/"dev27_gpu_qualification.json").read_text())
    if not cpu["result"]["passed"] or not old["passed"]:raise RuntimeError("unqualified source")
    for name,sha in cpu["source_sha256"].items():
        if n.digest(ROOT/name)!=sha:raise RuntimeError("CPU source changed")
    dependencies=["scripts/trr0020_stage27/single_position.py","scripts/trr0020_stage27/stable_cgls.py",
      "scripts/trr0020_stage21/linearized_prefix.py","scripts/trr0020_stage23/adjoint.py",
      "scripts/trr0020_stage24/normalized.py","scripts/trr0014/native.py","scripts/agent4/common.py",
      "src/token_reconstruction/public_prefix.py","scripts/trr0020_resource_guard.py"]
    for name in dependencies:
        if n.digest(ROOT/name)!=old["sources"][name]:raise RuntimeError("actual-prefix source changed")
    paths=list((ROOT/"scripts/trr0020_stage27").glob("*.py"))+[ROOT/v for v in dependencies]
    paths +=[X/v for v in ["DEV27_CHEAP_PLAN.md","dev27_cheap_preflight.json","dev27_cheap_cpu_reference.json","dev27_gpu_qualification.json"]]
    result={"task_id":"TRR-0020","kind":"public_cheap_current_token_directions","environment":n.environment(),
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv,
      "sources":{str(v.relative_to(ROOT)):n.digest(v) for v in paths},"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "start_unix":time.time(),"truth_read":False,"shortlist":None,"contexts":[],"cells":[],"passed":False,"forward_anchors":0,"cg1_anchors":0}
    def save(name,data):
        path=OUT/(name+".safetensors");t=time.perf_counter()
        n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(path))
        return {"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"save_and_hash_seconds":time.perf_counter()-t}
    try:
        n.guard();n.sync();t=time.perf_counter()
        prefix=n.inherited.load_prefix(torch.float32,asset_root=n.ASSETS/"backup");prefix.config._attn_implementation="eager"
        layers=public_parameters(prefix);radius=prefix.embed_tokens.weight.norm(dim=-1).median()
        n.sync();result["prefix_preparation_seconds"]=time.perf_counter()-t
        for length in [128,40]:
            context=next(v for v in old["contexts"] if v["length"]==length);source=context["source"]
            if n.digest(ROOT/source["path"])!=source["sha256"]:raise RuntimeError("public fixture changed")
            fixture=n.load_file(str(ROOT/source["path"]),device="cuda")
            base=prefix.embed_tokens(fixture["public_ids"]);target=fixture["public_target"];noisy=fixture["input"]
            pe=prefix.rotary_emb(base[None],torch.arange(length,device="cuda")[None]);cos,sin=pe[0][0],pe[1][0]
            selected=[length-1,length//2,1];snapshots={}
            past=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in layers]
            for pos in range(length):
                if pos in selected:snapshots[pos]=past
                _,_,current=forward(base[pos:pos+1],layers,past,cos[pos:pos+1],sin[pos:pos+1]);past=commit(past,current)
            for pos in selected:
                cur=noisy[pos:pos+1];history=snapshots[pos];co,si=cos[pos:pos+1],sin[pos:pos+1]
                raw,caches,_=forward(cur,layers,history,co,si)
                anchor=next(v for v in old["forwards"] if v["length"]==length and v["position"]==pos)
                if n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise RuntimeError("forward anchor changed")
                original=n.load_file(str(ROOT/anchor["path"]),device="cuda")
                equal=torch.equal(raw,original["actual"]) and torch.equal(cur,original["input"])
                result["contexts"].append({"length":length,"position":pos,"source":source,"forward_anchor":anchor,"equal":equal})
                if not equal:raise RuntimeError("current map changed")
                result["forward_anchors"]+=1
                rhs,mv,rmv=equation(raw,target[pos:pos+1],lambda v:jvp(v,layers,caches,co,si),lambda v:vjp(v,layers,caches,co,si))
                cg_reference,_=least_squares(mv,rmv,rhs,1,.0001)
                for rule in RULES:
                    first=None;reps=[]
                    for rep in range(3):
                        n.sync();t=time.perf_counter();delta,work=direction(rule,rhs,mv,rmv);n.sync();seconds=time.perf_counter()-t
                        factor=(radius/delta.norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
                        one,_,_=forward(cur+factor[:,None]*delta,layers,history,co,si)
                        half,_,_=forward(cur+.5*factor[:,None]*delta,layers,history,co,si)
                        target_unit=F.normalize(target[pos:pos+1],dim=-1)
                        one_error=F.normalize(one,dim=-1)-target_unit;half_error=F.normalize(half,dim=-1)-target_unit
                        data={"direction":delta,"linear_residual":mv(delta)-rhs,"one_step_residual":one_error,"half_step_residual":half_error,"clip_factor":factor}
                        if rule=="cg1":data["reference_cg1"]=cg_reference
                        artifact=save(f"{length}_{pos}_{rule}_{rep}",data)
                        equal=True if first is None else all(torch.equal(v,first[k]) for k,v in data.items())
                        if first is None:first={k:v.clone() for k,v in data.items()}
                        row={"rep":rep,"seconds":seconds,"repeat_equal":equal,"linear_relative_residual":float(data["linear_residual"].norm()/rhs.norm()),
                          "one_step_relative_residual":float(one_error.norm()/rhs.norm()),"half_step_relative_residual":float(half_error.norm()/rhs.norm()),**artifact}
                        reps.append(row)
                        if not equal or not all(torch.isfinite(v).all() for v in data.values()):raise RuntimeError("invalid output; preserved")
                        if rule=="cg1":
                            row["cg1_exact"]=torch.equal(delta,cg_reference)
                            if not row["cg1_exact"]:raise RuntimeError("one-step output changed")
                            result["cg1_anchors"]+=1
                        n.guard()
                    result["cells"].append({"length":length,"position":pos,"rule":rule,"repetitions":reps,**work,
                      "validation_jvp_calls_per_solve":1,"validation_forward_calls_per_solve":2})
                print("CHEAP_CURRENT_POSITION",length,pos,flush=True)
        if len(result["cells"])!=24 or result["forward_anchors"]!=6 or result["cg1_anchors"]!=18:raise RuntimeError("incomplete matrix")
        result["passed"]=True
    except Exception:
        result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("CHEAP_CURRENT_COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()

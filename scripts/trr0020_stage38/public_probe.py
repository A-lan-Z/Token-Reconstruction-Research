"""Public full-vocabulary BOS lookup qualification; never a shortlist."""
from pathlib import Path
import sys,os,json,time,subprocess,traceback,resource,gc
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"))
import native as n
from lookup import select
torch=n.torch;F=torch.nn.functional
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev38_public";DEST=X/"dev38_public_probe.json"
def guard():
    free,_=torch.cuda.mem_get_info()
    available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
    temp=int(subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],text=True))
    if free<3*2**30 or available<8*2**30 or torch.cuda.max_memory_reserved()>8*2**30 or temp>=80:raise RuntimeError("resource margin violated")
@torch.no_grad()
def main():
    if DEST.exists() or OUT.exists():raise RuntimeError("create-only")
    OUT.mkdir(parents=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("deterministic environment")
    prior=json.loads((X/"dev37_public_probe.json").read_text());stress=json.loads((X/"dev34_stress_probe.json").read_text())
    if not prior["passed"] or not stress["passed"]:raise RuntimeError("missing qualification")
    paths=list(Path(__file__).parent.glob("*.py"))+[ROOT/r for r in
      ["scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py","scripts/trr0020_resource_guard.py"]]
    paths +=[X/r for r in ["DEV38_PUBLIC_PLAN.md","dev38_public_preflight.json","dev37_public_probe.json","dev34_stress_probe.json"]]
    result={"task_id":"TRR-0020","scope":"explicitly public first-token numerical test, not reconstruction accuracy",
      "sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":sys.argv,"environment":n.environment(),"start_unix":time.time(),"truth_read":False,"shortlist":None,
      "passed":False,"families":[],"score_references":[],"preparation":{}}
    def save(name,data):
        p=OUT/(name+".safetensors");t=time.perf_counter();n.save_file({k:v.detach().contiguous().cpu() for k,v in data.items()},str(p))
        return {"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"io_hash_seconds":time.perf_counter()-t}
    try:
        guard();source=prior["table_source"];table_path=Path(source["path"])
        if n.digest(table_path)!=source["sha256"] or n.digest(n.ASSETS/"backup/prefix.safetensors")!=prior["prefix_sha256"]:raise RuntimeError("table/prefix changed")
        result["table_source"]=source;result["prefix_sha256"]=prior["prefix_sha256"];result["config_sha256"]=prior["config_sha256"]
        t=time.perf_counter();table=n.load_file(str(table_path),device="cuda")["response"];lengths=table.norm(dim=-1);n.sync()
        result["preparation"]["table_load_and_norm_seconds"]=time.perf_counter()-t
        fixture=stress["fixture"]
        if n.digest(ROOT/fixture["path"])!=fixture["sha256"]:raise RuntimeError("common fixture changed")
        common=n.load_file(str(ROOT/fixture["path"]));common_ids=common["public_ids"][:,1]
        generator=torch.Generator().manual_seed(38038)
        random_ids=torch.randint(0,len(table),(128,),generator=generator)
        result["random_seed"]=38038;result["common_fixture"]=fixture
        families=[]
        for precision,dtype,implementation in [("bf16",torch.bfloat16,"sdpa"),("fp32",torch.float32,"eager")]:
            n.sync();t=time.perf_counter();prefix=n.inherited.load_prefix(dtype,asset_root=n.ASSETS/"backup")
            prefix.config._attn_implementation=implementation;n.sync()
            result["preparation"][precision+"_prefix_load_seconds"]=time.perf_counter()-t
            t=time.perf_counter();targets=[]
            for token in random_ids:
                ids=torch.tensor([[128000,int(token)]],device="cuda")
                targets.append(prefix.forward_full(ids)[0,1].float().cpu())
            n.sync();result["preparation"][precision+"_public_target_generation_seconds"]=time.perf_counter()-t
            families.append((precision,"random",random_ids,torch.stack(targets)))
            families.append((precision,"common",common_ids,common[precision+"_target"][:,1].float()))
            del prefix;gc.collect();torch.cuda.empty_cache();guard()
        for precision,family,ids,observations in families:
            data={};seconds=[];first=None
            for rep in range(3):
                score_rows=[];tokens=[];times=[];transfer=0.
                for index,target in enumerate(observations):
                    h=target.to("cuda");n.sync();t=time.perf_counter()
                    chosen,scores=select(table,lengths,h);n.sync();times.append(time.perf_counter()-t)
                    t=time.perf_counter();score_rows.append(scores.cpu());tokens.append(chosen.cpu());transfer+=time.perf_counter()-t
                    if rep==0 and family=="common" and index<2:
                        reference=torch.cat([F.cosine_similarity(table[start:start+256],h[None],dim=-1,eps=1e-12) for start in range(0,len(table),256)])
                        torch.testing.assert_close(scores,reference,rtol=2e-5,atol=2e-6)
                        same=torch.equal(chosen,reference.argmax().reshape(1))
                        result["score_references"].append({"precision":precision,"index":index,"maximum_score_error":float((scores-reference).abs().max()),"argmax_equal":same})
                        if not same:raise RuntimeError("direct reference argmax differs")
                arrays={"scores":torch.stack(score_rows),"tokens":torch.cat(tokens),"public_ids":ids,"observations":observations}
                if not all(bool(torch.isfinite(v).all()) for v in arrays.values()):raise RuntimeError("nonfinite")
                equal=first is None or all(torch.equal(v,first[k]) for k,v in arrays.items())
                if first is None:first=arrays
                seconds.append({"rep":rep,"score_seconds":times,"full_score_transfer_seconds":transfer,"exact_repeat":equal})
                if not equal:raise RuntimeError("scores changed across repeats")
                guard()
            result["families"].append({"precision":precision,"family":family,"count":len(ids),"repetitions":seconds,**save(precision+"_"+family,first)})
            print("FULL_BOS_LOOKUP_COMPLETE",precision,family,len(ids),flush=True)
        if len(result["families"])!=4 or len(result["score_references"])!=4:raise RuntimeError("incomplete matrix")
        for path,sha in result["sources"].items():
            if n.digest(ROOT/path)!=sha:raise RuntimeError("source changed")
        result["passed"]=True
    except Exception:result["failure"]=traceback.format_exc();raise
    finally:
        n.sync();result.update(end_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
          peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        n.write(DEST,result);print("BOS_LOOKUP_PUBLIC_DONE",result["passed"],flush=True)
if __name__=="__main__":main()

from pathlib import Path
import sys,json,time,gc,subprocess,resource,os
ROOT=Path(__file__).resolve().parents[2]
for path in ["scripts/trr0014","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
from fast_soft import FastSoftVocabulary,BF16Mixture,fused_update
torch=n.torch;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":raise RuntimeError("required deterministic environment missing")
torch.use_deterministic_algorithms(True)
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev6_r1"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
CONFIGS=[(mode,steps) for mode in ["constant","cool","adaptive"] for steps in [128,256]]
def binding():
    paths=list((ROOT/"scripts/trr0020_stage6r1").glob("*.py"))+[X/"DEV6_R1_PLAN.md",
      ROOT/"scripts/trr0020_stage5/graph_soft.py",ROOT/"scripts/trr0020/soft_vocabulary.py",
      ROOT/"scripts/trr0020_stage3/reset_soft.py",ROOT/"scripts/trr0014/native.py",ROOT/"scripts/agent4/common.py",
      ROOT/"src/token_reconstruction/public_prefix.py",ROOT/"src/token_reconstruction/prefix_weight_metric.py"]
    return {"execution_flags":{"deterministic_algorithms":True,"cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"allow_tf32":False},"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in paths},
      "observations_sha256":n.digest(INPUT/"observations.safetensors"),
      "metadata_sha256":n.digest(INPUT/"metadata.json"),"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors")}
def kernel_tests(bound):
    torch.manual_seed(200026)
    shape=(3,257)
    z0=torch.randn(shape,device="cuda");m0=torch.randn(shape,device="cuda")*.001
    v0=torch.rand(shape,device="cuda")*.01;gradient=torch.randn(shape,device="cuda")*.0001
    error=torch.tensor([0.,.0125,.2],device="cuda")
    lr=torch.tensor(.6,device="cuda");decay=torch.tensor(.98,device="cuda");rows=[]
    for adaptive in [False,True]:
        z=z0.clone();m=m0.clone();v=v0.clone()
        fused_update(z,m,v,gradient,error,lr,decay,z.numel(),257,adaptive)
        mr=torch.lerp(m0,gradient,.1);vr=torch.lerp(v0,gradient.square(),.005)
        rate=lr*(error/.05).clamp(.01,1).sqrt()[:,None] if adaptive else lr
        zr=(z0-rate*mr/(vr.sqrt()+1e-12))*decay
        for actual,reference in [(z,zr),(m,mr),(v,vr)]:
            torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-7)
        rows.append({"adaptive":adaptive,"max_parameter_error":float((z-zr).abs().max())})
    probability=torch.softmax(torch.randn(4,513,device="cuda"),-1).detach().requires_grad_()
    weight=torch.randn(513,64,device="cuda").to(torch.bfloat16)
    gradient=torch.randn(4,64,device="cuda").to(torch.bfloat16).float()
    y=BF16Mixture.apply(probability,weight);y.backward(gradient)
    reference=(probability.detach().to(torch.bfloat16).cpu().double()@weight.cpu().double()).float()
    reference_gradient=(gradient.cpu().double()@weight.cpu().double().T).float()
    torch.testing.assert_close(y.cpu(),reference,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(probability.grad.cpu(),reference_gradient,rtol=2e-5,atol=2e-5)
    n.write(X/f"dev6_r1_kernel_tests_{time.time_ns()}.json",{"binding":bound,"optimizer":rows,
      "mixture_max_error":float((y.cpu()-reference).abs().max()),
      "mixture_gradient_max_error":float((probability.grad.cpu()-reference_gradient).abs().max()),
      "dtype":"BF16 operands FP32 output; independent CPU FP64 references","passed":True})
    print("Independent fused-update and mixture tests passed",flush=True)
def main():
    OUT.mkdir(parents=True,exist_ok=True);bound=binding();env=n.environment();start=time.time()
    kernel_tests(bound);n.guard()
    prefix=n.load_prefix();data=n.load_file(str(INPUT/"observations.safetensors"))
    selected=[];counts={}
    for row in json.loads((INPUT/"metadata.json").read_text()):
        key=(row["condition"],row["group"]);counts.setdefault(key,0)
        if counts[key]<2:selected.append(row);counts[key]+=1
    gen=torch.Generator().manual_seed(200026)
    fixture=torch.randint(256,128000,(1,128),generator=gen).to("cuda");fixture[0,0]=128000
    with torch.no_grad():h=prefix.forward_full(fixture)[0].float()
    entries=[];setups={}
    for mode,steps in CONFIGS:
        name=f"fast_{mode}{steps}"
        n.sync();t=time.perf_counter();engine=FastSoftVocabulary(prefix,mode,steps)
        n.sync();setups[name]={"engine_seconds":time.perf_counter()-t}
        runs=[];predictions=[]
        for repetition in range(3):
            current,current_stats=engine.decode(h);n.guard()
            raw=X/f"dev6_r1_qualification_{name}_{repetition}_{time.time_ns()}.safetensors"
            n.save_file(current,str(raw));predictions.append(current)
            runs.append({"path":str(raw.relative_to(ROOT)),"sha256":n.digest(raw),"stats":current_stats})
        output=predictions[0];stats=runs[0]["stats"];repeated_stats=runs[-1]["stats"]
        comparisons=[]
        for repetition in [1,2]:
            one=stats["loss_trace"];two=runs[repetition]["stats"]["loss_trace"]
            comparisons.append({"repetition":repetition,
                "tokens_equal":all(torch.equal(output[k],predictions[repetition][k]) for k in output),
                "losses_equal":one==two,"maximum_loss_difference":max(abs(a-b) for a,b in zip(one,two)),
                "different_loss_entries":sum(a!=b for a,b in zip(one,two))})
        audit={"binding":bound,"runs":runs,"comparisons":comparisons,"environment":env,"setup":setups[name]}
        n.write(X/f"dev6_r1_repeatability_{name}_{time.time_ns()}.json",audit)
        if not all(c["tokens_equal"] for c in comparisons):raise RuntimeError("nonrepeatable largest geometry; traces preserved")
        if not all(c["losses_equal"] for c in comparisons):raise RuntimeError("nonrepeatable optimization path; traces preserved")
        qualification={"stats":stats,"repeat_stats":repeated_stats,"repeatability":comparisons,
          "synthetic_correct":{k:int((v[1:]==fixture[0,1:].cpu()).sum()) for k,v in output.items()},
          "repeated_outputs_and_loss_trace_equal":True}
        free,total=torch.cuda.mem_get_info()
        if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError("insufficient qualified margin")
        n.write(X/f"dev6_r1_qualification_{name}_{time.time_ns()}.json",{"binding":bound,"qualification":qualification,
          "setup":setups[name],"environment":env,"free_bytes":free,
          "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
        print("qualified",name,round(repeated_stats["total_seconds"],3),qualification["synthetic_correct"],flush=True)
        for row in selected:
            path=OUT/(row["id"]+"__"+name+".safetensors");rp=path.with_suffix(".json")
            if rp.exists():
                e=json.loads(rp.read_text())
                if e["binding"]!=bound or n.digest(path)!=e["sha256"]:raise RuntimeError("changed resume")
                entries.append(e);continue
            if path.exists():raise RuntimeError("orphan output")
            out,stats=engine.decode(data[row["id"]]);n.guard()
            n.save_file(out,str(path))
            e={**row,"method":name,"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),
               "binding":bound,"stats":stats,"frozen_unix":time.time()}
            n.write(rp,e);entries.append(e);print("frozen",row["id"],name,round(stats["total_seconds"],3),flush=True)
        setups[name]["capture_events"]=engine.capture_events
        del engine;gc.collect();torch.cuda.empty_cache()
    if binding()!=bound:raise RuntimeError("source changed")
    assert len(entries)==48
    n.write(X/"dev6_r1_freeze.json",{"task_id":"TRR-0020","binding":bound,"environment":env,"entries":entries,
       "selected_records":selected,"configurations":CONFIGS,"setups":setups,"truth_read":False,
       "start_unix":start,"end_unix":time.time(),"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
       "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
       "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"command":sys.argv})
    print("COMPLETE48CELLFREEZE",flush=True)
if __name__=="__main__":main()

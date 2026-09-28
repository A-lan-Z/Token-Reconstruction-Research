from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1]
for path in ["scripts/trr0014","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5","scripts/trr0020_stage6"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
from fast_soft import FastSoftVocabulary
torch=n.torch
p=argparse.ArgumentParser();p.add_argument("--deterministic",action="store_true");args=p.parse_args()
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(args.deterministic)
X=ROOT/"experiments/TRR-0020";tag="deterministic" if args.deterministic else "default"
prefix=n.load_prefix();fixture=X/"numerics_public_fixture.safetensors"
if args.deterministic:
    data=n.load_file(str(fixture));h=data["observation"].to("cuda");ids=data["ids"]
else:
    generator=torch.Generator().manual_seed(200026)
    ids=torch.randint(256,128000,(1,128),generator=generator);ids[0,0]=128000
    with torch.no_grad():h=prefix.forward_full(ids.to("cuda"))[0].float()
    n.save_file({"observation":h.cpu(),"ids":ids},str(fixture))
engine=FastSoftVocabulary(prefix,"constant",256);runs=[];predictions=[]
for rep in range(3):
    out,stats=engine.decode(h);n.guard()
    path=X/f"numerics_{tag}_{rep}.safetensors";n.save_file(out,str(path))
    runs.append({"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),"stats":stats})
    predictions.append(out);print(tag,rep,round(stats["total_seconds"],3),flush=True)
differences=[]
for i in [1,2]:
    a=runs[0]["stats"]["loss_trace"];b=runs[i]["stats"]["loss_trace"]
    diffs=[k for k,(v,w) in enumerate(zip(a,b)) if v!=w]
    differences.append({"repeat":i,"different_loss_entries":len(diffs),
      "maximum_loss_difference":max(abs(v-w) for v,w in zip(a,b)),
      "first_difference":None if not diffs else [diffs[0],a[diffs[0]],b[diffs[0]]],
      "token_output_equal":{k:bool(torch.equal(predictions[0][k],predictions[i][k])) for k in predictions[0]}})
n.write(X/f"numerics_{tag}.json",{"environment":n.environment(),"deterministic":args.deterministic,
 "workspace_config":__import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
 "fixture_sha256":n.digest(fixture),"source_sha256":n.digest(Path(__file__)),
 "implementation_sha256":n.digest(ROOT/"scripts/trr0020_stage6/fast_soft.py"),
 "runs":runs,"differences":differences,"peak_allocated":torch.cuda.max_memory_allocated(),
 "peak_reserved":torch.cuda.max_memory_reserved(),"scope":"public paired128-token repeatability diagnosis"})
print(json.dumps(differences,indent=2),flush=True)

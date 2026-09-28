from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5","scripts/trr0020_stage6r1","scripts/trr0020_stage7"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
from epsilon_soft import EpsilonVocabulary,CONFIGS
from discrete_soft import DiscreteSoftVocabulary
torch=n.torch;torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev14"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
TRACE_KEYS=["loss_trace","observed_error_trace","mean_confidence_trace","mean_gini_trace"]
def binding():
    paths=list((ROOT/"scripts/trr0020_stage14").glob("*.py"))+[X/"DEV14_PLAN.md",X/"dev14_scaled_kernel_reference.json",X/"dev7_freeze.json"]
    paths +=[ROOT/p for p in ["scripts/trr0020_stage7/discrete_soft.py","scripts/trr0020_stage6r1/fast_soft.py",
      "scripts/trr0020_stage5/graph_soft.py","scripts/trr0020_stage3/reset_soft.py","scripts/trr0020/soft_vocabulary.py",
      "scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py",
      "src/token_reconstruction/prefix_weight_metric.py","scripts/trr0020_resource_guard.py"]]
    return {"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in sorted(paths)},
      "observations_sha256":n.digest(INPUT/"observations.safetensors"),"metadata_sha256":n.digest(INPUT/"metadata.json"),
      "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "execution_flags":{"deterministic_algorithms":True,"cublas_workspace_config":":4096:8","allow_tf32":False}}
def rows():
    selected=[];counts={}
    for r in json.loads((INPUT/"metadata.json").read_text()):
        key=r["condition"],r["group"];counts.setdefault(key,0)
        if counts[key]<2:selected.append(r);counts[key]+=1
    assert len(selected)==8
    return selected
def equal(a,sa,b,sb,keys=None):
    keys=set(a) if keys is None else set(keys)
    return all(torch.equal(a[k],b[k]) for k in keys) and all(sa[k]==sb[k][:len(sa[k])] for k in TRACE_KEYS)
def verify(e,bound):
    if e["binding"]!=bound or n.digest(ROOT/e["path"])!=e["sha256"]:raise ValueError("changed cell")
    return n.load_file(str(ROOT/e["path"]))

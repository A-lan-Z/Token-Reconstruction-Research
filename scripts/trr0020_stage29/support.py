from pathlib import Path
import sys,os,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5",
             "scripts/trr0020_stage6r1","scripts/trr0020_stage7","scripts/trr0020_stage8","scripts/trr0017",
             "scripts/trr0020_stage10r1","scripts/trr0020_stage15","scripts/trr0020_canonical_refine"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
from mirror_optimizer import MirrorOptimizer,CONFIGS,CHECKPOINTS,AUX_KEYS,gradient_reference
torch=n.torch;torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev29_grid"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
TRACE_KEYS=["loss_trace","observed_error_trace","mean_confidence_trace","mean_gini_trace"]
def guard():
    free,total=torch.cuda.mem_get_info()
    available=int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))*1024
    temp=int(subprocess.check_output(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],text=True).strip())
    if free<3*2**30 or available<8*2**30 or torch.cuda.max_memory_reserved()>8*2**30 or temp>=80:
        raise RuntimeError(f"dev29 resource limit:free={free};host={available};peak={torch.cuda.max_memory_reserved()};temperature={temp}")
def binding():
    paths=list((ROOT/"scripts/trr0020_stage29").glob("*.py"))
    paths +=[X/name for name in ["DEV29_PROSPECTIVE.md","DEV29_GRID_PLAN.md","dev29_preflight.json",
      "dev29_cpu_reference.json","dev7_freeze.json","dev26_grid_freeze.json"]]
    paths +=[ROOT/name for name in ["scripts/trr0020_canonical_refine/reuse_stream.py","scripts/trr0020_stage15/warm_refinement.py",
      "scripts/trr0020_stage10r1/proximal.py","scripts/trr0017/discrete_parallel.py","scripts/trr0017/joint_projection.py",
      "scripts/trr0020_stage8/natural_soft.py","scripts/trr0020_stage7/discrete_soft.py","scripts/trr0020_stage6r1/fast_soft.py",
      "scripts/trr0020_stage5/graph_soft.py","scripts/trr0020_stage3/reset_soft.py","scripts/trr0020/soft_vocabulary.py",
      "scripts/trr0014/native.py","scripts/agent4/common.py","src/token_reconstruction/public_prefix.py",
      "src/token_reconstruction/prefix_weight_metric.py","scripts/trr0020_resource_guard.py"]]
    return {"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in sorted(paths)},
      "observations_sha256":n.digest(INPUT/"observations.safetensors"),"metadata_sha256":n.digest(INPUT/"metadata.json"),
      "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),"config_sha256":n.digest(n.ASSETS/"backup/config.json"),
      "execution_flags":{"deterministic_algorithms":True,"cublas_workspace_config":":4096:8","allow_tf32":False}}
def rows():
    result=[];counts={}
    for row in json.loads((INPUT/"metadata.json").read_text()):
        key=row["condition"],row["group"];counts.setdefault(key,0)
        if counts[key]<2:result.append(row);counts[key]+=1
    assert len(result)==8
    return result
def equal(a,sa,b,sb):
    return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a) and all(sa["warm"][k]==sb["warm"][k] for k in TRACE_KEYS) and sa["mirror_traces"]==sb["mirror_traces"]
def verify(entry,bound):
    if entry["binding"]!=bound or n.digest(ROOT/entry["path"])!=entry["sha256"]:raise ValueError("changed cell")
    return n.load_file(str(ROOT/entry["path"]))

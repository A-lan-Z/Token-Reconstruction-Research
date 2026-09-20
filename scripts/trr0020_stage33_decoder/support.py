from pathlib import Path
import sys,os,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
from decoder import CausalDecoder,CONFIGS,AUX_KEYS,n,torch
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage33_replay"))
from qualify import guard
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("determinism flags")
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev33_decoder"
INPUT=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1"
def binding():
    qualification=json.loads((X/"dev33_replay_qualification.json").read_text())
    if not qualification["passed"]:raise RuntimeError("unqualified replay")
    paths=list((ROOT/"scripts/trr0020_stage33_decoder").glob("*.py"))
    paths +=[X/name for name in ["DEV33_DECODER_PLAN.md","dev33_decoder_preflight.json","dev33_replay_qualification.json"]]
    sources=qualification["sources"]|{str(p.relative_to(ROOT)):n.digest(p) for p in paths}
    for name,sha in sources.items():
        if n.digest(ROOT/name)!=sha:raise RuntimeError("qualified source changed "+name)
    return {"sources":sources,"observations_sha256":n.digest(INPUT/"observations.safetensors"),
      "metadata_sha256":n.digest(INPUT/"metadata.json"),"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "config_sha256":n.digest(n.ASSETS/"backup/config.json"),
      "flags":{"TF32":False,"deterministic":True,"CUBLAS_WORKSPACE_CONFIG":":4096:8"}}
def rows():
    result=[];counts={}
    for row in json.loads((INPUT/"metadata.json").read_text()):
        key=row["condition"],row["group"];counts.setdefault(key,0)
        if counts[key]<2:result.append(row);counts[key]+=1
    if len(result)!=8:raise RuntimeError("wrong panel")
    return result
def equal(a,b):return set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items())
def verify(entry,bound):
    if entry["binding"]!=bound or n.digest(ROOT/entry["path"])!=entry["sha256"]:raise ValueError("changed output")
    return n.load_file(str(ROOT/entry["path"]))

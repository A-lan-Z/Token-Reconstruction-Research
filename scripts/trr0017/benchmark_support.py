from pathlib import Path
import sys,json,hashlib,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT.parent/"TRR-0015"
X=ROOT/"experiments/TRR-0017";OUT=ROOT/"outputs/TRR-0017/benchmark"
INPUT=PREVIOUS/"outputs/TRR-0015"
sys.path.insert(0,str(ROOT/"scripts/trr0014"));import native as n
sys.path.insert(0,str(ROOT/"scripts/trr0017"))
torch=n.torch;write=n.write;digest=n.digest
METHODS=("a1a2","fragment_native","fragment_shared","continuous96","discrete64")
CANONICAL=("clean-pile-lora-64x40","historical-finance-strict-bos-128x128")
def binding():
    paths=sorted((ROOT/"scripts/trr0017").glob("*.py"))
    paths += [X/"BENCHMARK_PLAN.md",X/"registry.json",
         ROOT/"scripts/trr0014/fragment_predict.py",ROOT/"scripts/trr0014/native.py",
         ROOT/"src/token_reconstruction/public_prefix.py",
         ROOT/"src/token_reconstruction/prefix_fragments.py",
         ROOT/"src/token_reconstruction/prefix_weight_metric.py",
         ROOT/"src/token_reconstruction/a1a2_configuration_search.py"]
    return {"source_hashes":{str(p.relative_to(ROOT)):digest(p) for p in paths},
            "observations_sha256":digest(INPUT/"observations.safetensors"),
            "metadata_sha256":digest(INPUT/"metadata.json"),
            "prefix_sha256":digest(n.ASSETS/"backup/prefix.safetensors")}
def utc():return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def verify_cell(entry):
    p=ROOT/entry["path"]
    if digest(p)!=entry["sha256"]:raise ValueError("changed prediction")
    data=n.load_file(str(p));tokens=data["tokens"]
    if list(tokens.shape)!=[entry["positions"]] or tokens.dtype!=torch.int64 or int(tokens[0])!=128000:
        raise ValueError("invalid token output")
    if not bool(((tokens>=0)&(tokens<128256)).all()):raise ValueError("invalid token ID")
    if entry["method"] in METHODS[:3]:
        if list(data["candidates"].shape)!=[len(tokens),256]:raise ValueError("invalid candidate geometry")
        selected=data["candidates"][1:].gather(1,data["scores"].argmax(-1,keepdim=True)).squeeze(1)
        if not torch.equal(tokens[1:],selected):raise ValueError("wrong direct-cosine choice")
        if not all(bool(torch.isfinite(data[k]).all()) for k in ("scores","mse")):raise ValueError("nonfinite")
    return data


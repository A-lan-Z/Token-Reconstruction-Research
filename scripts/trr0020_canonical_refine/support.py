from pathlib import Path
import sys,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0014","scripts/trr0017","scripts/trr0019","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5","scripts/trr0020_stage6r1","scripts/trr0020_stage7","scripts/trr0020_stage10r1","scripts/trr0020_stage15"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
torch=n.torch
X=ROOT/"experiments/TRR-0020/canonical_refine";OUT=ROOT/"outputs/TRR-0020/canonical_refine"
INPUT=ROOT.parent/"TRR-0018/outputs/TRR-0018"
METHODS=("a1_native","warm128white003","a1_graph")
CANONICAL=("clean-pile-lora-64x40","historical-finance-strict-bos-128x128")
METHOD_IDS={"warm128white003":"no_shortlist_warm128_quadratic_white003_positionbest32"}
def utc():return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def binding():
    paths=list((ROOT/"scripts/trr0020_canonical_refine").glob("*.py"))+list((ROOT/"scripts/trr0019").glob("*.py"))
    paths += list((ROOT/"scripts/trr0020_stage15").glob("*.py"))+[ROOT/"scripts/trr0020_stage10r1/proximal.py",ROOT/"scripts/trr0017/discrete_parallel.py",ROOT/"scripts/trr0017/joint_projection.py"]
    paths += [X/p for p in ["BENCHMARK_PLAN.md","registry.json","metadata.json","inherited_registry.json","inherited_matrix.json"]]
    paths += [ROOT/p for p in ["scripts/trr0014/native.py","scripts/trr0014/fragment_predict.py","scripts/agent4/comparator.py","scripts/agent4/common.py",
      "scripts/trr0019/replay_a2.py","scripts/trr0017/shared_context.py","scripts/trr0020/soft_vocabulary.py","scripts/trr0020_stage3/reset_soft.py",
      "scripts/trr0020_stage5/graph_soft.py","scripts/trr0020_stage6r1/fast_soft.py","scripts/trr0020_stage7/discrete_soft.py",
      "src/token_reconstruction/public_prefix.py","src/token_reconstruction/prefix_weight_metric.py","src/token_reconstruction/component_crossover.py",
      "src/token_reconstruction/a1a2_configuration_search.py"]]
    return {"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in sorted(set(paths))},
      "observations_sha256":n.digest(INPUT/"observations.safetensors"),"prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
      "baseline_lens_sha256":n.digest(n.ASSETS/"backup/lens_alpaca.pt")}
def tensor_hashes(result):
    return {k:hashlib.sha256(v.contiguous().cpu().numpy().tobytes()).hexdigest() for k,v in sorted(result.items())}
def verify(entry):
    p=ROOT/entry["path"]
    if n.digest(p)!=entry["sha256"]:raise ValueError("changed output")
    data=n.load_file(str(p));t=entry["positions"]
    shapes={"tokens":(t,)} if entry["method"] in METHOD_IDS else {"tokens":(t,),"candidates":(t,256),"scores":(t-1,256),"mse":(t-1,256)}
    if set(data)!=set(shapes) or any(tuple(data[k].shape)!=shape for k,shape in shapes.items()):raise ValueError("wrong geometry")
    if data["tokens"].dtype!=torch.int64 or int(data["tokens"][0])!=128000:raise ValueError("wrong tokens/BOS")
    if tensor_hashes(data)!=entry["tensor_sha256"]:raise ValueError("changed tensor values")
    if entry["method"] in METHOD_IDS:
        if entry["stats"]["shortlist_size"] is not None or entry["stats"]["separate_candidate_verification_calls"]!=0:raise ValueError("forbidden shortlist")
    else:
        choice=data["candidates"][1:].gather(1,data["scores"].argmax(-1,keepdim=True)).flatten()
        if not torch.equal(choice,data["tokens"][1:]):raise ValueError("baseline decision mismatch")
        if not all(torch.isfinite(data[k]).all() for k in ["scores","mse"]):raise ValueError("nonfinite baseline score")
    return data
def phase_gate(f,bound):
    if f["binding"]!=bound:raise ValueError("phase binding changed")
    rows=json.loads((X/"metadata.json").read_text())
    if {(e["id"],e["method"],e["rep"]) for e in f["entries"]}!={(r["id"],f["method"],f["rep"]) for r in rows}:raise ValueError("phase matrix incomplete")
    if len(f["entries"])!=len(rows):raise ValueError("duplicate phase cells")
    for e in f["entries"]:
        if e["binding"]!=bound:raise ValueError("cell binding changed")
        verify(e)

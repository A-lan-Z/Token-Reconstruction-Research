from pathlib import Path
import sys,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0019"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage50"))
import grid50_support as dev
n=dev.n;torch=n.torch
X=ROOT/"experiments/TRR-0020/canonical_moment";OUT=ROOT/"outputs/TRR-0020/canonical_moment"
INPUT=ROOT.parent/"TRR-0018/outputs/TRR-0018"
METHODS=("a1_native","moment64","a1_graph")
CANONICAL=("clean-pile-lora-64x40","historical-finance-strict-bos-128x128")
METHOD_IDS={"moment64":"no_shortlist_bennett64_positionbest"}
def utc():return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def binding():
    frozen=json.loads((X/"development_binding.json").read_text())
    if dev.binding()!=frozen:raise ValueError("development implementation binding changed")
    paths=list(Path(__file__).parent.glob("*.py"))+list((ROOT/"scripts/trr0019").glob("*.py"))
    paths += [X/name for name in ["BENCHMARK_PLAN.md","registry.json","metadata.json","inherited_registry.json","inherited_matrix.json","development_binding.json","preflight.json"]]
    paths += [ROOT/name for name in ["RESEARCH_CHARTER.md","research/DUAL_BENCHMARK_PROTOCOL.md",
      "scripts/trr0014/native.py","scripts/trr0014/fragment_predict.py","scripts/agent4/comparator.py","scripts/agent4/common.py",
      "src/token_reconstruction/public_prefix.py","src/token_reconstruction/component_crossover.py",
      "src/token_reconstruction/a1a2_configuration_search.py","scripts/trr0020_resource_guard.py",
      "experiments/TRR-0020/dev50_phase_bennett.json","experiments/TRR-0020/dev50_freeze.json","experiments/TRR-0020/dev50_score.json"]]
    return {"sources":{str(p.relative_to(ROOT)):n.digest(p) for p in sorted(set(paths))},
      "observations_sha256":n.digest(INPUT/"observations.safetensors"),
      "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
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
        stats=entry["stats"]
        if not stats["numeric_valid"] or stats["mirror_steps"]!=64 or stats["factor"]!=2. or stats["moment_rule"]!="bennett" or stats["moment_root_iterations"]!=48 or stats["model_parameter_updates"]!=0 or stats["vocabulary_entries_per_sweep"]!=128256:raise ValueError("selected rule changed")
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

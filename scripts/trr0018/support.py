"""Shared immutable artifacts for TRR-0018; no evaluator truth access."""
from pathlib import Path
import sys, json, time, hashlib, subprocess
ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parent.parent
PREVIOUS = ROOT.parent / "TRR-0014"
X = ROOT / "experiments/TRR-0018"
OUT = ROOT / "outputs/TRR-0018"
METHODS = ("mixed256", "frequency256", "a1a2")
BUDGETS = {m:256 for m in METHODS}
CANONICAL = ("clean-pile-lora-64x40", "historical-finance-strict-bos-128x128")
sys.path.insert(0, str(ROOT / "scripts/trr0014"))
import native as n
torch = n.torch
load_file, save_file = n.load_file, n.save_file

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2)

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def source_hashes():
    paths = list((ROOT / "scripts/trr0018").glob("*.py"))
    paths += [
        ROOT / "src/token_reconstruction/prefix_fragments.py",
        ROOT / "src/token_reconstruction/prefix_fragment_frequency.py",
        ROOT / "src/token_reconstruction/prefix_fragment_mixed.py",
        ROOT / "src/token_reconstruction/prefix_weight_metric.py",
        ROOT / "src/token_reconstruction/a1a2_configuration_search.py",
        ROOT / "src/token_reconstruction/component_crossover.py",
        ROOT / "src/token_reconstruction/public_prefix.py",
        ROOT / "src/token_reconstruction/historical_inputlens_bridge.py",
        ROOT / "scripts/trr0014/fragment_predict.py",
        ROOT / "scripts/trr0014/native.py",
        ROOT / "scripts/agent4/comparator.py",
        ROOT / "scripts/agent4/common.py",
        X / "PLAN.md", X / "BENCHMARK_PLAN.md", X / "registry.json",
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}

def binding():
    return {
        "implementation_hashes": source_hashes(),
        "observation_sha256": digest(OUT / "observations.safetensors"),
        "metadata_sha256": digest(OUT / "metadata.json"),
        "prefix_sha256": digest(n.ASSETS / "backup/prefix.safetensors"),
        "a1_lens_sha256": digest(n.ASSETS / "backup/lens_alpaca.pt"),
    }

def check_binding(value):
    if value != binding():
        raise ValueError("implementation, inputs, assets, or registry changed")

def verify_cell(entry, row):
    p = ROOT / entry["path"]
    if digest(p) != entry["sha256"]:
        raise ValueError("prediction changed")
    data = load_file(str(p))
    length, k = row["positions"], BUDGETS[entry["method"]]
    expected = {"tokens": (length,), "candidates": (length,k),
                "scores": (length-1,k), "mse": (length-1,k)}
    if set(data) != set(expected) or any(tuple(data[key].shape) != shape for key,shape in expected.items()):
        raise ValueError("wrong prediction geometry")
    if data["tokens"].dtype != torch.int64 or data["candidates"].dtype != torch.int64:
        raise ValueError("wrong token dtype")
    if int(data["tokens"][0]) != 128000:
        raise ValueError("wrong BOS")
    for tensor in (data["tokens"], data["candidates"][1:]):
        if bool(((tensor < 0) | (tensor >= 128256)).any()):
            raise ValueError("invalid token")
    if not all(bool(torch.isfinite(data[key]).all()) for key in ("scores","mse")):
        raise ValueError("nonfinite verification")
    selected = data["candidates"][1:].gather(1, data["scores"].argmax(-1,keepdim=True)).squeeze(1)
    if not torch.equal(selected,data["tokens"][1:]):
        raise ValueError("decision rule changed")
    if len(entry["phases"]) != 3 or any(p["total"] <= 0 for p in entry["phases"]):
        raise ValueError("missing repeated timing")
    if entry["logical_simulations"] != (length-1)*k:
        raise ValueError("wrong candidate cost")
    return data

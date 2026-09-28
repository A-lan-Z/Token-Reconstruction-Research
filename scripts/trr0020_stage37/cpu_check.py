"""Independent materialized-vector checks of direct full-table scoring."""
from pathlib import Path
import sys,json,time,subprocess,hashlib
import torch
from torch.nn import functional as F
from direct_scores import norms_for_shift,scores
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/"experiments/TRR-0020/dev37_cpu_reference.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows=[];start=time.time()
    for dtype in [torch.float32,torch.float64]:
      for kind in ["ordinary","exact_cancel","near_cancel","zero_target"]:
        generator=torch.Generator().manual_seed(37037)
        table=torch.randn(37,11,generator=generator,dtype=dtype)
        shift=torch.randn(11,generator=generator,dtype=dtype)
        target=torch.randn(11,generator=generator,dtype=dtype)
        if kind=="exact_cancel":table[5]=-shift
        if kind=="near_cancel":table[5]=-shift+1e-5*torch.randn(11,generator=generator,dtype=dtype)
        if kind=="zero_target":target.zero_()
        square,indices=norms_for_shift(table,table.square().sum(-1),shift)
        cosine,distance=scores(table,square,shift,target,indices)
        explicit=table+shift
        reference_cosine=F.cosine_similarity(explicit,target[None],dim=-1,eps=1e-12)
        reference_distance=-(explicit-target).square().sum(-1)
        # Cancellation can affect the expanded numerator too. Use the same
        # explicit row only for ill-conditioned arithmetic in the implementation.
        torch.testing.assert_close(square,explicit.square().sum(-1),rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(distance,reference_distance,rtol=2e-6,atol=2e-5)
        torch.testing.assert_close(cosine,reference_cosine,rtol=2e-5,atol=2e-5)
        rows.append({"dtype":str(dtype),"case":kind,"norm_fallback_rows":len(indices),
          "maximum_cosine_error":float((cosine-reference_cosine).abs().max()),
          "maximum_distance_error":float((distance-reference_distance).abs().max()),
          "argmax_equal":bool(cosine.argmax()==reference_cosine.argmax() and distance.argmax()==reference_distance.argmax()),
          "passed":True})
    result={"task_id":"TRR-0020","kind":"CPU full-table score algebra","passed":all(r["argmax_equal"] for r in rows),
      "rows":rows,"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":sys.argv,"start_unix":start,"end_unix":time.time(),
      "environment":{"python":sys.version,"torch":torch.__version__},
      "sources":{str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    with DEST.open("x") as f:json.dump(result,f,indent=2)
    if not result["passed"]:raise RuntimeError("argmax disagreement")
    print(json.dumps(result))
if __name__=="__main__":main()

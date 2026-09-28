"""Byte-exact torch refactoring of unchanged reductions and scalar rules."""
from pathlib import Path
import sys,json,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage32"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage44"))
sys.path.insert(0,str(Path(__file__).parent))
import torch
from budget_step import update as original
from reused_step import update
from pointwise import TorchPointwise,bits_equal
DEST=ROOT/"experiments/TRR-0020/dev46_cpu_reference.json"
def fixture(vocab,kind,dtype=torch.float32):
    g=torch.Generator().manual_seed(44044+vocab)
    logits=torch.randn(3,vocab,generator=g,dtype=dtype)*(50 if kind=="peaked" else 2)
    gradient=torch.randn(3,vocab,generator=g,dtype=dtype)
    if kind=="flat":gradient.fill_(.5)
    error=torch.tensor([0.,.02,.4],dtype=dtype)
    if kind=="zero_budget":error.zero_()
    return logits,gradient,error
def main():
    if DEST.exists():raise RuntimeError("create-only")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    paths=list(Path(__file__).parent.glob("*.py"))+list((ROOT/"scripts/trr0020_stage44").glob("*.py"))+[ROOT/"scripts/trr0020_stage32/budget_step.py",ROOT/"experiments/TRR-0020/DEV46_PLAN.md"]
    result={"task_id":"TRR-0020","scope":"byte-exact CPUrefactoring only; GPUkernels unqualified",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"environment":{"torch":torch.__version__,"python":sys.version},
      "sources":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
      "start_unix":time.time(),"passed":False,"cases":[]}
    try:
      for dtype in [torch.float32,torch.float64]:
       for vocab in [17,1023,1031]:
        for kind in ["random","peaked","flat","zero_budget"]:
            logits,g,e=fixture(vocab,kind,dtype)
            expected,info=original(logits,g,e,2.,4);probability=torch.softmax(logits,dim=-1)
            before=probability.clone()
            got,details=update(logits,g,e,2.,4,pw=TorchPointwise,probability=probability)
            exact=bits_equal(probability,before) and bits_equal(got,expected) and all(bits_equal(details[k],v) for k,v in info.items())
            result["cases"].append({"dtype":str(dtype),"vocab":vocab,"kind":kind,"all_bytes_equal":exact,
              "maximum_output_error":float((got-expected).abs().max()),
              "differing_fields":[k for k,v in info.items() if not bits_equal(details[k],v)]})
            if not exact:raise RuntimeError("CPUrefactoring changed bytes")
      result["passed"]=len(result["cases"])==24
    except Exception:
      import traceback
      result["failure"]=traceback.format_exc();raise
    finally:
      result["end_unix"]=time.time()
      with DEST.open("x") as z:json.dump(result,z,indent=2)
      print(json.dumps({k:v for k,v in result.items() if k not in ["sources","cases"]}))
if __name__=="__main__":main()

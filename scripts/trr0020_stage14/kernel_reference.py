"""Small GPU reference for Adam denominator behavior, saved before any gate."""
from pathlib import Path
import sys,json,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]
for path in ["src","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5","scripts/trr0020_stage6r1","scripts/trr0020_stage7"]:
    sys.path.insert(0,str(ROOT/path))
import torch,triton
from epsilon_soft import epsilon_update_kernel
from fast_soft import fused_update
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
N=111;VOCAB=37
levels=torch.tensor([0.,1e-5,-1e-5,1e-10,-1e-10,1e-14,-1e-14,1e-18,-1e-18,1e-20,-1e-20,1e-23,-1e-23],device="cuda")
g=levels[torch.arange(N,device="cuda")%len(levels)]
error=torch.tensor([.01,.05,.1],device="cuda");lr=torch.tensor(.6,device="cuda");decay=torch.tensor(.98,device="cuda")
results=[]
for eps in [1e-12,1e-16,1e-20]:
    z=torch.zeros(N,device="cuda");m=torch.zeros_like(z);v=torch.zeros_like(z)
    epsilon_update_kernel[(triton.cdiv(N,1024),)](z,m,v,g,error,lr,decay,N,VOCAB,eps,1024,num_warps=4,enable_fp_fusion=False)
    gd=g.double();rate=lr.double()*torch.sqrt((error.double()[torch.arange(N,device="cuda")//VOCAB]/.05).clamp(.01,1.))
    reference=-rate*(.1*gd)/torch.sqrt(.005*gd.square()).add(eps)*decay.double()
    diff=(z.double()-reference).abs();tolerance=2e-6+5e-5*reference.abs()
    cases=[]
    for i,level in enumerate(levels.cpu().tolist()):
        selected=(torch.arange(N,device="cuda")%len(levels))==i
        cases.append({"gradient":level,"max_abs_error":float(diff[selected].max()),"max_update":float(z[selected].abs().max())})
    results.append({"epsilon":eps,"max_abs_error":float(diff.max()),"within_declared_tolerance":bool((diff<=tolerance).all()),
      "cases":cases,"finite":bool(torch.isfinite(z).all())})
    if eps==1e-12:
        native=torch.zeros_like(z);nm=torch.zeros_like(z);nv=torch.zeros_like(z)
        fused_update(native,nm,nv,g,error,lr,decay,N,VOCAB,True)
        results[-1]["original_kernel_byte_equal"]=bool(torch.equal(z,native) and torch.equal(m,nm) and torch.equal(v,nv))
free,total=torch.cuda.mem_get_info()
record={"scope":"111-coordinate GPU numerical test; no reconstruction data","utc_unix":time.time(),
 "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
 "kernel_source_sha256":hashlib.sha256((ROOT/"scripts/trr0020_stage14/epsilon_soft.py").read_bytes()).hexdigest(),
 "hardware":torch.cuda.get_device_name(),"torch":torch.__version__,"command":sys.argv,"results":results,
 "free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved(),"rtol":5e-5,"atol":2e-6}
name=sys.argv[1] if len(sys.argv)>1 else "dev14_kernel_reference.json"
with (ROOT/"experiments/TRR-0020"/name).open("x") as f:json.dump(record,f,indent=2)
print(json.dumps(record,indent=2))

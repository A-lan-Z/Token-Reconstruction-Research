"""Profile a public synthetic optimizer step; no evaluated observations or labels."""
from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
for path in ["scripts/trr0014","scripts/trr0020","scripts/trr0020_stage3","scripts/trr0020_stage5"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
from graph_soft import GraphSoftVocabulary
torch=n.torch
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
X=ROOT/"experiments/TRR-0020"
prefix=n.load_prefix();engine=GraphSoftVocabulary(prefix,"cosine",.6,True)
gen=torch.Generator().manual_seed(200026)
ids=torch.randint(256,128000,(1,128),generator=gen).to("cuda");ids[0,0]=128000
with torch.no_grad():h=prefix.forward_full(ids)[0].float()
engine.ensure(128);engine.reset(h);torch.backends.cuda.matmul.allow_tf32=True
for i in range(3):engine.train_step(128)
n.sync();n.guard()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True) as prof:
    for i in range(5):engine.train_step(128)
    n.sync()
n.guard()
rows=[]
for event in prof.key_averages(group_by_input_shape=True):
    device=getattr(event,"self_device_time_total",0.)
    if device:
        rows.append({"operation":event.key,"count":event.count,"self_gpu_us":device,"input_shapes":event.input_shapes})
rows.sort(key=lambda e:-e["self_gpu_us"])
prof.export_chrome_trace(str(X/"optimizer_profile_trace.json"))
n.write(X/"optimizer_profile.json",{"environment":n.environment(),"seed":200026,"length":128,
 "steps":5,"rows":rows,"table":prof.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total",row_limit=20),
 "source_sha256":n.digest(Path(__file__)),"implementation_sha256":n.digest(ROOT/"scripts/trr0020_stage5/graph_soft.py"),
 "prefix_sha256":n.digest(n.ASSETS/"backup/prefix.safetensors"),
 "trace_sha256":n.digest(X/"optimizer_profile_trace.json"),"command":sys.argv,
 "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
 "scope":"public synthetic eager training-step GPU operator profile; not reconstruction accuracy or production latency"})
print(json.dumps(rows[:15],indent=2),flush=True)

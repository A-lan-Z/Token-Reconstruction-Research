"""Truth-free prediction CLI, create-only per-record outputs and final receipt."""
from common import *
from solver import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--dtype',default='bfloat16');p.add_argument('--run',required=True);p.add_argument('--settings');p.add_argument('--method',choices=['prefix','a1a2'],default='prefix');p.add_argument('--device',default='cuda');args=p.parse_args()
torch.set_num_threads(2)
start=time.perf_counter();env=environment();prefix=load_prefix(getattr(torch,args.dtype),device=args.device);guard() if args.device=="cuda" else None
inputs=OUT/args.panel/args.dtype
obs=load_file(str(inputs/'observations.safetensors'))
meta=json.loads((inputs/'metadata.json').read_text())
settings=Settings(**json.loads(Path(args.settings).read_text())) if args.settings else Settings()
run=EVID/args.run;run.mkdir(exist_ok=False)
if args.method=='a1a2':
    from comparator import prepare,decode
    lens,embedding=prepare(prefix)
load_seconds=time.perf_counter()-start
torch.cuda.reset_peak_memory_stats() if args.device=="cuda" else None
# Public synthetic warmup, never scored.
with torch.no_grad():prefix.forward_full(torch.tensor([[128000,1]],device=args.device))
paths=[]
for r in meta:
    guard() if args.device=="cuda" else None
    result=reconstruct(prefix,obs[r['id']],settings,guard=guard if args.device=="cuda" else None) if args.method=='prefix' else decode(prefix,obs[r['id']],lens,embedding,guard=guard if args.device=="cuda" else None)
    result.update({'record_id':r['id'],'group':r['group'],'environment':env})
    path=run/f"{r['id']}.json";write(path,result);paths.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
    print(r['id'],result['seconds'],flush=True)
write(run/'freeze.json',{'method':args.method,'settings':asdict(settings) if args.method=='prefix' else {'K':256,'selector':'direct_cosine'},'environment':env,'end_utc':environment()['utc'],'prediction_files':paths,'observations_sha256':digest(inputs/'observations.safetensors'),'load_seconds':load_seconds,'total_process_seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated() if args.device=="cuda" else 0,'peak_reserved':torch.cuda.max_memory_reserved() if args.device=="cuda" else 0,'truth_opened':False})

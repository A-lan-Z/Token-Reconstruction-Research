"""Prediction-only immutable runner; no source IDs/text are read."""
from shared import *
from safetensors.torch import load_file
import argparse,resource
p=argparse.ArgumentParser();p.add_argument('--panel',default='old');p.add_argument('--method',required=True);p.add_argument('--name',required=True);args=p.parse_args()
torch.set_num_threads(2)
start=time.perf_counter();env=environment();prefix=load_prefix(torch.bfloat16);guard()
root=OLD/'gpu_retrospective/bfloat16' if args.panel=='old' else OUT/args.panel
obs=load_file(str(root/'observations.safetensors'));meta=json.loads((root/'metadata.json').read_text())
if args.method in ('original','profile'):
    if args.method=='profile':
        from profiled import reconstruct,Settings
    else:
        from solver import reconstruct,Settings
    settings=Settings(**json.loads((ROOT/'experiments/agent4-prefix-only-inversion/adam_settings.json').read_text()))
    decode=lambda h:reconstruct(prefix,h,settings,guard=guard)
elif args.method=='a1a2':
    from comparator import prepare,decode as compare
    lens,embedding=prepare(prefix)
    decode=lambda h:compare(prefix,h,lens,embedding,guard=guard)
else:
    from search import reconstruct
    decode=lambda h:reconstruct(prefix,h,args.method,guard=guard)
setup=time.perf_counter()-start
dst=E/args.name;dst.mkdir(exist_ok=False)
with torch.no_grad():prefix.forward_full(torch.tensor([[128000,1]],device='cuda'))
sync();torch.cuda.reset_peak_memory_stats()
paths=[]
for row in meta:
    result=decode(obs[row['id']]);result.update(record_id=row['id'],group=row['group'],environment=env)
    path=dst/(row['id']+'.json');write(path,result);paths.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
    print(row['id'],result['seconds'],flush=True)
sync()
write(dst/'freeze.json',{'method':args.method,'panel':args.panel,'environment':env,'end_utc':environment()['utc'],'prediction_files':paths,'observations_sha256':digest(root/'observations.safetensors'),'setup_seconds':setup,'process_seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'truth_read':False})

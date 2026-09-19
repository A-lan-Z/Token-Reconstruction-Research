from graph_probe import *
from replay_a2 import ReplayA2
import argparse,gc

def compare(a,b):
    return {k:{'equal':torch.equal(a[k],b[k]),'different':int((a[k]!=b[k]).sum()),
       'max_abs':float((a[k].float()-b[k].float()).abs().max())} for k in a}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fast',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=n.load_prefix();bos=torch.tensor([[128000]],device='cuda');engine=ReplayA2(p,128,a.fast)
    n.guard();inp=ROOT.parent/'TRR-0015/outputs/TRR-0015';obs=n.load_file(str(inp/'observations.safetensors'))
    rows=json.loads((inp/'metadata.json').read_text());records=[]
    # Ordered128,40,variable,128 checks restarting at BOS after short sequences.
    choices=[rows[0],next(x for x in rows if x['positions']==40),next(x for x in rows if x['positions']==43),rows[1]]
    for row in choices:
        archived=n.load_file(str(inp/'predictions'/(row['id']+'__fragment256.safetensors')))
        h=obs[row['id']].to('cuda').float();ids=archived['candidates'].to('cuda')
        reference=execute(p,h,ids,bos,False);n.sync();times=[];first=None
        for rep in range(3):
            n.sync();t=time.perf_counter();raw=engine.run(h,ids);n.sync();times.append(time.perf_counter()-t)
            result={k:v.cpu() for k,v in raw.items()}
            if first is None:first=result
            elif not all(torch.equal(first[k],result[k]) for k in first):raise RuntimeError('nonrepeatable replay')
        dif=compare(first,{k:v.cpu() for k,v in reference.items()})
        if not a.fast:assert all(v['equal'] for v in dif.values()),dif
        records.append({'id':row['id'],'positions':len(h),'times':times,'differences':dif})
    # In-place public-weight refresh must reach the captured operations.
    parameter=p.layers[0].mlp.down_proj.weight;saved=parameter.detach().clone()
    with torch.no_grad():parameter.add_(torch.tensor(0.001,device=parameter.device,dtype=parameter.dtype))
    n.sync();reference=execute(p,h[:8],ids[:8],bos,False);raw=engine.run(h[:8],ids[:8]);n.sync()
    update_dif=compare(raw,reference)
    if not a.fast:assert all(v['equal'] for v in update_dif.values()),update_dif
    with torch.no_grad():parameter.copy_(saved)
    raw=engine.run(h[:8],ids[:8]);n.sync();restored={k:v.cpu() for k,v in raw.items()}
    reference=execute(p,h[:8],ids[:8],bos,False)
    restore_dif=compare(restored,{k:v.cpu() for k,v in reference.items()})
    if not a.fast:assert all(v['equal'] for v in restore_dif.values()),restore_dif
    receipt={'task_id':'TRR-0019','fast':a.fast,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_hashes':{str(x.relative_to(ROOT)):n.digest(x) for x in (ROOT/'scripts/trr0019').glob('*.py')},
      'command':sys.argv,'environment':n.environment(),'truth_read':False,'setup':engine.setup,'records':records,
      'weight_refresh':update_dif,'weight_restore':restore_dif,'peak_allocated':torch.cuda.max_memory_allocated(),
      'peak_reserved':torch.cuda.max_memory_reserved(),'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    n.write(X/('replay_fast_qualification.json' if a.fast else 'replay_exact_qualification.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()

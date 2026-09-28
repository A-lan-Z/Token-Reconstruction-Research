from support import *
from torch.nn import functional as F
import subprocess,resource

def main():
    OUT.mkdir(parents=True,exist_ok=True);bound=binding();env=n.environment();started=time.time();n.guard()
    n.sync();start=time.perf_counter();prefix=n.load_prefix();n.sync();preparation=time.perf_counter()-start
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200045)).to('cuda');fixture[0,0]=128000
    qualification=[]
    for length in [128,40]:
        first=None
        for rep in range(3):
            n.sync();begin=time.perf_counter();h=prefix.forward_full(fixture[:,:length]).float().cpu();n.sync()
            path=OUT/f'qualification_{length}_{rep}_{time.time_ns()}.safetensors';n.save_file({'activation':h},str(path))
            if first is None:first=h
            qualification.append({'length':length,'rep':rep,'path':str(path.relative_to(ROOT)),'sha256':n.digest(path),'equal':torch.equal(h,first),'seconds':time.perf_counter()-begin,'peak_reserved':torch.cuda.max_memory_reserved()})
    n.write(X/f'qualification_{time.time_ns()}.json',{'binding':bound,'runs':qualification,'environment':env})
    if not all(q['equal'] for q in qualification) or torch.cuda.max_memory_reserved()>4*2**30:raise RuntimeError('qualification failed; saved')
    print('QUALIFIED_NATIVE128',round(torch.cuda.max_memory_reserved()/2**30,4),flush=True)
    observed=n.load_file(str(INPUT/'observations.safetensors'));entries=[]
    for index,old in enumerate(selected()):
        path=OUT/f"{old['id']}__{old['method']}.safetensors";receipt=path.with_suffix('.json')
        if receipt.exists():
            e=json.loads(receipt.read_text())
            if e['binding']!=bound or n.digest(path)!=e['sha256']:raise ValueError('changed checkpoint')
            entries.append(e);continue
        if path.exists():raise ValueError('orphan diagnostic')
        ids=original(old);n.sync();begin=time.perf_counter()
        predicted=prefix.forward_full(ids.to('cuda')[None])[0].float();target=observed[old['id']].to('cuda').float()
        data={'tokens':ids,'cosine_error':(1-F.cosine_similarity(predicted[1:],target[1:],dim=-1)).cpu(),'mse':(predicted[1:]-target[1:]).square().mean(-1).cpu()}
        n.sync();seconds=time.perf_counter()-begin;n.save_file(data,str(path));free,total=torch.cuda.mem_get_info()
        e={k:old[k] for k in ['id','setup_id','method','positions','source_id']}
        e.update(path=str(path.relative_to(ROOT)),sha256=n.digest(path),original_path=old['path'],original_sha256=old['sha256'],binding=bound,seconds=seconds,frozen_unix=time.time(),peak_reserved=torch.cuda.max_memory_reserved(),free_bytes=free)
        n.write(receipt,e);entries.append(e);n.guard()
        if free<2*2**30 or torch.cuda.max_memory_reserved()>4*2**30:raise RuntimeError('resource margin; output saved')
        if index%128==0:print('FROZEN',index+1,flush=True)
    f={'task_id':'TRR-0020','kind':'diagnostic_only_no_reconstruction_changes','binding':bound,'entries':entries,'qualification':qualification,'environment':env,'prefix_preparation_seconds':preparation,'start_unix':started,'end_unix':time.time(),'truth_read':False,'peak_reserved':torch.cuda.max_memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'command':sys.argv}
    gate(f);n.write(X/'freeze.json',f);print('COMPLETE768DIAGNOSTIC_FREEZE',flush=True)
if __name__=='__main__':main()

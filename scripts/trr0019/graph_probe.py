from pathlib import Path
import sys,time,json,subprocess,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/trr0014'));import native as n
sys.path.insert(0,str(ROOT/'scripts/trr0017'));from shared_context import shared_candidates
import torch
from torch.nn import functional as F
X=ROOT/'experiments/TRR-0019'

@torch.inference_mode()
def execute(prefix,h,ids,bos,shared=True):
    cache=prefix.new_cache();prefix.run_cached(bos,cache,0)
    tokens=[bos.reshape(())];scores=[];mses=[]
    for pos in range(1,len(h)):
        response=(shared_candidates if shared else n.candidates)(prefix,cache,ids[pos],pos).float()
        score=F.cosine_similarity(response,h[pos:pos+1],dim=-1)
        mse=(response-h[pos]).square().mean(-1)
        winner=ids[pos].gather(0,score.argmax().reshape(1))
        tokens.append(winner.reshape(()));scores.append(score);mses.append(mse)
        prefix.run_cached(winner.reshape(1,1),cache,pos)
    return {'tokens':torch.stack(tokens),'candidates':ids,'scores':torch.stack(scores),'mse':torch.stack(mses)}

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=n.load_prefix();bos=torch.tensor([[128000]],device='cuda')
    inp=ROOT.parent/'TRR-0015/outputs/TRR-0015'
    obs=n.load_file(str(inp/'observations.safetensors'));rows=json.loads((inp/'metadata.json').read_text())
    row=next(x for x in rows if x['positions']==128)
    archived=n.load_file(str(inp/'predictions'/(row['id']+'__fragment256.safetensors')))
    h=obs[row['id']].to('cuda').float();ids=archived['candidates'].to('cuda')
    reference=execute(p,h,ids,bos,False);n.sync()
    comparisons={k:torch.equal(reference[k].cpu(),archived[k]) for k in reference}
    assert all(comparisons.values()),comparisons
    start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):warm=execute(p,h,ids,bos)
    torch.cuda.current_stream().wait_stream(stream);n.sync();del warm
    warm_end=time.perf_counter();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):result=execute(p,h,ids,bos)
    n.sync();capture_end=time.perf_counter();n.guard()
    measurements={k:[] for k in ['native','shared','graph']};equal=[]
    for rep in range(5):
        for method in list(measurements)[rep%3:]+list(measurements)[:rep%3]:
            n.sync();t=time.perf_counter()
            if method=='graph':graph.replay();out=result
            else:out=execute(p,h,ids,bos,method=='shared')
            n.sync();measurements[method].append(time.perf_counter()-t)
            eq={k:torch.equal(out[k],reference[k]) for k in reference};equal.append({'method':method,**eq})
            assert all(eq.values()),eq
    receipt={'task_id':'TRR-0019','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_sha256':n.digest(Path(__file__)),'command':sys.argv,'environment':n.environment(),'input_id':row['id'],
      'observations_sha256':n.digest(inp/'observations.safetensors'),'truth_read':False,'archived_output_equal':comparisons,
      'warmup_seconds':warm_end-start,'capture_seconds':capture_end-warm_end,'measurements':measurements,'equal':equal,
      'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
      'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    n.write(X/'monolithic_qualification.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()

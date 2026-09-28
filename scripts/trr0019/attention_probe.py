from graph_probe import *
import graph_probe
from shared_attention import fast_candidates

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=n.load_prefix();bos=torch.tensor([[128000]],device='cuda')
    inp=ROOT.parent/'TRR-0015/outputs/TRR-0015'
    obs=n.load_file(str(inp/'observations.safetensors'));rows=json.loads((inp/'metadata.json').read_text())
    row=next(x for x in rows if x['positions']==128)
    archived=n.load_file(str(inp/'predictions'/(row['id']+'__fragment256.safetensors')))
    h=obs[row['id']].to('cuda').float();ids=archived['candidates'].to('cuda')
    reference=execute(p,h,ids,bos,False);n.sync()
    graph_probe.shared_candidates=fast_candidates
    start=time.perf_counter();fast=execute(p,h,ids,bos);n.sync();compile_end=time.perf_counter()
    differences={k:{'equal':torch.equal(fast[k],reference[k]),'differing':int((fast[k]!=reference[k]).sum()),
        'max_abs':float((fast[k].float()-reference[k].float()).abs().max())} for k in fast}
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):warm=execute(p,h,ids,bos)
    torch.cuda.current_stream().wait_stream(stream);n.sync();del warm
    graph=torch.cuda.CUDAGraph();capture_start=time.perf_counter()
    with torch.cuda.graph(graph,stream=stream):result=execute(p,h,ids,bos)
    n.sync();capture_end=time.perf_counter();n.guard()
    measurements={'native':[],'direct_shared_attention':[],'graph_shared_attention':[]}
    for rep in range(5):
        for method in list(measurements)[rep%3:]+list(measurements)[:rep%3]:
            n.sync();t=time.perf_counter()
            if method=='graph_shared_attention':graph.replay();out=result
            else:out=execute(p,h,ids,bos,method!='native')
            n.sync();measurements[method].append(time.perf_counter()-t)
            expected=reference if method=='native' else fast
            assert all(torch.equal(out[k],expected[k]) for k in out),'nonrepeatable'
    receipt={'task_id':'TRR-0019','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_hashes':{str(x.relative_to(ROOT)):n.digest(x) for x in (ROOT/'scripts/trr0019').glob('*.py')},
      'command':sys.argv,'environment':n.environment(),'input_id':row['id'],'truth_read':False,
      'compile_and_first_run_seconds':compile_end-start,'capture_seconds':capture_end-capture_start,
      'measurements':measurements,'differences':differences,
      'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
      'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    n.write(X/'attention_qualification.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()

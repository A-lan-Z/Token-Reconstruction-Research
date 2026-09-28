from support import *
import argparse,subprocess,resource

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    name,steps=CONFIGS[args.index];bound=binding();phase=X/f'dev19_phase_{name}.json'
    if phase.exists():
        f=json.loads(phase.read_text())
        if f['binding']!=bound:raise RuntimeError('changed phase')
        for e in f['entries']:verify(e,bound)
        return
    OUT.mkdir(parents=True,exist_ok=True);env=n.environment();started=time.time();n.guard()
    cpu=json.loads((X/'dev19_cpu_reference.json').read_text())
    if not cpu['passed'] or cpu['source_sha256']!=n.digest(ROOT/'scripts/trr0020_stage19/cpu_reference.py'):raise RuntimeError('unqualified algebra')
    n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();prefix_seconds=time.perf_counter()-t
    t=time.perf_counter();engine=MeanReadout(prefix,steps);n.sync();engine_seconds=time.perf_counter()-t
    fixture=torch.randint(256,128000,(1,128),generator=torch.Generator().manual_seed(200041)).to('cuda');fixture[0,0]=128000
    runs=[];controls=[];references=[]
    for length in [128,40]:
        h=prefix.forward_full(fixture[:,:length])[0].float();first=None
        for rep in range(3):
            output,stats=engine.decode(h);n.guard()
            path=X/f'dev19_qualification_{name}_{length}_{rep}_{time.time_ns()}.safetensors';n.save_file(output,str(path))
            if first is None:first,first_stats=output,stats
            free,total=torch.cuda.mem_get_info()
            runs.append({'length':length,'path':str(path.relative_to(ROOT)),'sha256':n.digest(path),'stats':stats,'outputs_and_traces_equal':equal(output,stats,first,first_stats),'free_bytes':free,'peak_reserved':torch.cuda.max_memory_reserved()})
        reference,rs=engine.soft.decode(h)
        path=X/f'dev19_qualification_control_{name}_{length}_{time.time_ns()}.safetensors';n.save_file(reference,str(path))
        controls.append({'length':length,'path':str(path.relative_to(ROOT)),'sha256':n.digest(path),'tokens_equal':all(torch.equal(first['warm_'+k],v) for k,v in reference.items()),'traces_equal':all(first_stats['soft'][k]==rs[k] for k in TRACE_KEYS)})
        references.append({'length':length,**score_reference(engine,first['fitted_embedding'])})
    q={'binding':bound,'runs':runs,'controls':controls,'score_reference':references,'environment':env,'peak_reserved':torch.cuda.max_memory_reserved(),'prefix_seconds':prefix_seconds,'engine_seconds':engine_seconds}
    n.write(X/f'dev19_qualification_{name}_{time.time_ns()}.json',q)
    if not all(v['outputs_and_traces_equal'] and v['stats']['readout_numeric_valid'] for v in runs):raise RuntimeError('nonrepeatable or nonfinite; saved')
    if not all(v['tokens_equal'] and v['traces_equal'] for v in controls):raise RuntimeError('changed soft reference; saved')
    if not all(v['all_close'] for v in references):raise RuntimeError('score formula failed; saved')
    if min(v['free_bytes'] for v in runs)<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError('memory margin; saved')
    print('QUALIFIED',name,round(torch.cuda.max_memory_reserved()/2**30,4),flush=True)
    old={e['id']:e for e in json.loads((X/'dev7_freeze.json').read_text())['entries'] if e['method']=='gini003'}
    observed=n.load_file(str(INPUT/'observations.safetensors'));entries=[]
    for row in rows():
        path=OUT/f"{row['id']}__{name}.safetensors";receipt=path.with_suffix('.json')
        if receipt.exists():
            e=json.loads(receipt.read_text());verify(e,bound);entries.append(e);continue
        if path.exists():raise RuntimeError('orphan output')
        output,stats=engine.decode(observed[row['id']]);n.guard();n.save_file(output,str(path))
        anchor=old[row['id']]
        if n.digest(ROOT/anchor['path'])!=anchor['sha256']:raise RuntimeError('changed anchor')
        previous=n.load_file(str(ROOT/anchor['path']));keys=['step'+str(v) for v in [0,16,32,64,128,256] if v<=steps]
        same=all(torch.equal(output['warm_'+k],previous[k]) for k in keys) and all(stats['soft'][k]==anchor['stats'][k][:steps+1] for k in TRACE_KEYS)
        free,total=torch.cuda.mem_get_info();e={**row,'method':name,'path':str(path.relative_to(ROOT)),'sha256':n.digest(path),'binding':bound,'stats':stats,'frozen_unix':time.time(),'peak_reserved':torch.cuda.max_memory_reserved(),'free_bytes':free,'original_anchor':{'path':anchor['path'],'sha256':anchor['sha256'],'snapshots_and_trace_prefixes_equal':same}}
        n.write(receipt,e)
        if not same or not stats['readout_numeric_valid']:raise RuntimeError('invalid cell; saved')
        if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError('cell margin; saved')
        entries.append(e);print('FROZEN',row['id'],name,round(stats['total_seconds'],4),flush=True)
    if binding()!=bound:raise RuntimeError('source changed')
    n.write(phase,{'task_id':'TRR-0020','binding':bound,'method':name,'configuration':CONFIGS[args.index],'entries':entries,'qualification':q,'environment':env,'start_unix':started,'end_unix':time.time(),'truth_read':False,'prefix_seconds':prefix_seconds,'engine_seconds':engine_seconds,'capture_events':engine.capture_events,'peak_reserved':torch.cuda.max_memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'command':sys.argv})
    print('PHASE_COMPLETE',name,flush=True)
if __name__=='__main__':main()

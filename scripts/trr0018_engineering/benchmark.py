"""Post-confirmation engineering study; preserve every frozen output value."""
from pathlib import Path
import sys,json,time,hashlib,argparse,resource
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts/trr0018'))
from support import *
sys.path.insert(0,str(R/'scripts/trr0018'))
import predict as original
from token_reconstruction.shared_candidate_cache import shared_candidates
VARIANTS=('mixed_native','mixed_shared','a1a2_native')

def tensor_hashes(data):
    return {k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()} for k,v in data.items()}

def identity(data,reference):
    assert set(data)==set(reference)
    assert all(torch.equal(data[k],reference[k]) for k in data)
    a=tensor_hashes(data);assert a==tensor_hashes(reference);return a

def decode(prefix,h,variant,proposer,lens,emb):
    prior=original.inherited.candidates
    try:
        if variant=='mixed_shared':original.inherited.candidates=shared_candidates
        return original.decode(prefix,h,'a1a2' if variant=='a1a2_native' else 'mixed256',proposer,lens,emb)
    finally:original.inherited.candidates=prior

def engineering_binding():
    files=[Path(__file__),ROOT/'src/token_reconstruction/shared_candidate_cache.py',X/'ENGINEERING_PLAN.md']
    return {'original':binding(),'additional_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files},
            'frozen_reference_receipt_sha256':digest(X/'prediction_receipt.json')}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualify',action='store_true');args=parser.parse_args()
    bind=engineering_binding();prefix,proposer,lens,emb,setup=original.setup()
    if args.qualify:
        with torch.no_grad():h=prefix.forward_full(torch.tensor([[128000]+list(range(1000,1127))],device='cuda'))[0].cpu()
        results={};times={}
        for variant in VARIANTS:
            times[variant]=[]
            for repeat in range(3):
                result,phases=decode(prefix,h,variant,proposer,lens,emb);times[variant].append(phases)
                if repeat==0:results[variant]=result
                else:identity(result,results[variant])
            n.guard()
        identity(results['mixed_shared'],results['mixed_native'])
        cache=n.new_context(prefix);ids=results['mixed_native']['candidates'].to('cuda');tokens=results['mixed_native']['tokens'].to('cuda')
        for pos in range(1,128):
            before=[(layer.keys.clone(),layer.values.clone()) for layer in cache.backend.layers]
            a=n.candidates(prefix,cache,ids[pos],pos).float();b=shared_candidates(prefix,cache,ids[pos],pos)
            assert torch.equal(a,b),pos
            assert cache.length==pos
            for layer,(keys,values) in zip(cache.backend.layers,before):
                assert torch.equal(keys,layer.keys) and torch.equal(values,layer.values)
            prefix.run_cached(tokens[pos].reshape(1,1),cache,pos);n.guard()
        write(X/'engineering_qualification.json',{'binding':bind,'setup':setup,'largest_repetitions':times,
            'all127_candidate_hidden_arrays_equal':True,'committed_caches_unchanged_by_trials':True,
            'mixed_full_outputs_identical':True,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'utc':utc()})
        print('All127 candidate arrays and repeated full outputs identical',flush=True);return
    q=json.loads((X/'engineering_qualification.json').read_text());assert q['binding']==bind
    rows=json.loads((OUT/'metadata.json').read_text());observations=load_file(str(OUT/'observations.safetensors'))
    golden=json.loads((X/'prediction_receipt.json').read_text());references={(e['id'],e['method']):e for e in golden['entries']}
    dst=OUT/'engineering_receipts';dst.mkdir(exist_ok=True);entries=[]
    session=X/f'sessions/engineering_{time.time_ns()}.json';write(session,{'binding':bind,'setup':setup,'utc':utc(),'command':sys.argv})
    for i,row in enumerate(rows):
        for variant in VARIANTS[i%3:]+VARIANTS[:i%3]:
            path=dst/(row['id']+'__'+variant+'.json')
            if path.exists():
                entry=json.loads(path.read_text());assert entry['binding']==bind;entries.append(entry);continue
            n.guard();data,phases=decode(prefix,observations[row['id']],variant,proposer,lens,emb)
            reference=references[(row['id'],'a1a2' if variant=='a1a2_native' else 'mixed256')]
            before=time.perf_counter();p=ROOT/reference['path'];assert digest(p)==reference['sha256']
            values=load_file(str(p));hashes=identity(data,values)
            entry={**row,'variant':variant,'binding':bind,'phases':phases,'seconds':phases['total'],
                   'reference_path':reference['path'],'reference_sha256':reference['sha256'],'computed_tensor_hashes':hashes,
                   'all_arrays_byte_identical':True,'equivalence_and_io_seconds':time.perf_counter()-before,
                   'logical_simulations':(row['positions']-1)*256,'session':str(session.relative_to(ROOT)),'frozen_utc':utc()}
            write(path,entry);entries.append(entry)
        if (i+1)%8==0:print(f'{i+1}/{len(rows)} paired engineering cells complete',flush=True)
    assert engineering_binding()==bind
    write(X/'engineering_receipt.json',{'binding':bind,'setup':setup,'entries':entries,'observations':len(rows),'cells':len(entries),
          'all_shared_and_native_outputs_identical_to_frozen_reference':True,'timed_invocations_per_cell':1,
          'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
          'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'completed_utc':utc(),
          'accuracy_status':'retrospective execution equivalence; original R4 accuracy freeze remains governing; no truth read by this process'})
    print('Complete1296-cell output-equivalent engineering timing matrix',flush=True)
if __name__=='__main__':main()

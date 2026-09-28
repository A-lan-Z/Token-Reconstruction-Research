from benchmark_support import *
import resource
from replay_a2 import ReplayA2
from transformers import AutoTokenizer
from token_reconstruction.prefix_fragments import PrefixFragmentProposer,build_suffix_cache
from token_reconstruction.prefix_fragment_mixed import MixedFragmentView
from token_reconstruction.component_crossover import propose_public_a1
from comparator import prepare
import fragment_predict as inherited

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bound=binding();env=n.environment();started=utc();setup={}
    for p in [OUT,OUT/'predictions',OUT/'receipts']:p.mkdir(parents=True,exist_ok=True)
    n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();setup['prefix_load']=time.perf_counter()-t
    t=time.perf_counter();tok=AutoTokenizer.from_pretrained(n.ASSETS/'backup',local_files_only=True)
    suffix=build_suffix_cache(tok,len(prefix.embed_tokens.weight));setup['tokenizer_suffix']=time.perf_counter()-t
    n.sync();t=time.perf_counter();base=PrefixFragmentProposer(prefix,suffix);setup['table_stats']=base.build(256)
    n.sync();setup['prefix_table_rebuild']=time.perf_counter()-t;proposer=MixedFragmentView(base)
    t=time.perf_counter();lens,emb=prepare(prefix);n.sync();setup['a1_asset_load']=time.perf_counter()-t
    exact=ReplayA2(prefix);setup['exact_replay']=exact.setup
    fast=ReplayA2(prefix,fast=True);setup['fast_replay']=fast.setup;n.guard()
    def decode(h,method):
        if method.endswith('_native'):
            return inherited.decode(prefix,h,'a1a2' if method.startswith('a1') else 'fragment512',proposer,lens,emb)
        engine=fast if method=='mixed_fast' else exact
        n.sync();start=time.perf_counter();h=h.to('cuda').float()
        if method.startswith('a1'):
            prop=propose_public_a1(observations=h.unsqueeze(0),attention_mask=torch.ones((1,len(h)),dtype=torch.long),lens=lens,normalized_embeddings=emb)
            ids=prop.candidates[0,:,:256].to('cuda')
        else:ids=proposer.propose(h)
        n.sync();proposed=time.perf_counter();result=engine.run(h,ids);n.sync();verified=time.perf_counter()
        output={k:v.cpu() for k,v in result.items()};n.sync();end=time.perf_counter()
        return output,{'total':end-start,'proposal_and_input':proposed-start,'verification_and_commit':verified-proposed,'output_transfer':end-verified}
    rows=json.loads((X/'metadata.json').read_text());observations=n.load_file(str(INPUT/'observations.safetensors'))
    qual=next(r for r in rows if r['positions']==128);qresults={};qtimes={}
    for m in METHODS:
        reps=[];first=None
        for rep in range(3):
            result,phases=decode(observations[qual['id']],m);n.guard();reps.append(phases)
            if first is None:first=result
            elif not all(torch.equal(first[k],result[k]) for k in first):raise RuntimeError('nonrepeatable largestcell')
        qresults[m]=first;qtimes[m]=reps
    for a,b in [('a1_native','a1_graph'),('mixed_native','mixed_graph')]:
        if not all(torch.equal(qresults[a][k],qresults[b][k]) for k in qresults[a]):raise RuntimeError('exact replay qualification failed')
    free,total=torch.cuda.mem_get_info()
    if free<3*2**30 or torch.cuda.max_memory_reserved()>8*2**30:raise RuntimeError('combined qualification insufficient headroom')
    qpath=X/('benchmark_qualification_'+str(time.time_ns())+'.json')
    write(qpath,{'binding':bound,'environment':env,'setup':setup,'id':qual['id'],'phases':qtimes,
         'exact_graph_equal':True,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'free_bytes':free})
    print('Combined largestcell passed; freeGiB',round(free/2**30,3),'reservedGiB',round(torch.cuda.max_memory_reserved()/2**30,3),flush=True)
    entries=[]
    for i,row in enumerate(rows):
        order=METHODS[i%len(METHODS):]+METHODS[:i%len(METHODS)]
        for m in order:
            path=OUT/'predictions'/(row['id']+'__'+m+'.safetensors');rp=OUT/'receipts'/(row['id']+'__'+m+'.json')
            if rp.exists():
                e=json.loads(rp.read_text())
                if e['binding']!=bound:raise ValueError('resume binding changed')
                verify(e);entries.append(e);continue
            if path.exists():raise ValueError('orphan output: '+str(path))
            phases=[];first=None
            for rep in range(3):
                n.guard();result,timing=decode(observations[row['id']],m);phases.append(timing)
                if first is None:first=result
                elif not all(torch.equal(first[k],result[k]) for k in first):raise RuntimeError('nonrepeatable cell')
            reproduction=None
            if m!='mixed_fast':
                oldm='a1a2' if m.startswith('a1') else 'mixed256'
                prior=INPUT/'predictions'/(row['id']+'__'+oldm+'.safetensors');old=n.load_file(str(prior))
                reproduction={k:bool(torch.equal(first[k],old[k])) for k in first}
                if not all(reproduction.values()):raise RuntimeError('archived output changed: '+str(reproduction))
            t=time.perf_counter();n.save_file(first,str(path));sha=digest(path);io=time.perf_counter()-t
            e={**row,'method':m,'path':str(path.relative_to(ROOT)),'sha256':sha,'binding':bound,'phases':phases,
              'median_seconds':sorted(v['total'] for v in phases)[1],'repetitions_identical':True,'archived_reproduction':reproduction,
              'logical_simulations':(row['positions']-1)*256,'io_hash_seconds':io,'frozen_utc':utc()}
            verify(e);write(rp,e);entries.append(e)
        print(f'{i+1}/{len(rows)}',row['id'],'frozen',flush=True)
    if binding()!=bound:raise RuntimeError('source changed during run')
    write(X/'prediction_freeze.json',{'task_id':'TRR-0019','binding':bound,'environment':env,'setup':setup,
       'start_utc':started,'end_utc':utc(),'entries':entries,'truth_read':False,'qualification_path':str(qpath.relative_to(ROOT)),
       'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
       'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'command':sys.argv})
    print('FULL1360CELLFREEZE COMPLETE',flush=True)
if __name__=='__main__':main()

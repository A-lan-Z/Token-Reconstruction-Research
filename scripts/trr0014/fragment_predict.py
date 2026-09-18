"""Common native reconstruction and timing for the frozen fragment policy."""
from native import *
from token_reconstruction.prefix_fragments import PrefixFragmentProposer,build_suffix_cache,suffix_ids,intrinsic_table
from token_reconstruction.component_crossover import propose_public_a1
from torch.nn import functional as F
from transformers import AutoTokenizer
from comparator import prepare,decode as original_comparator
import argparse,resource
METHODS=['fragment512','union128','a1a2']
@torch.no_grad()
def decode(prefix,h,method,proposer,lens,emb):
    sync();start=time.perf_counter();h=h.to('cuda').float()
    if method=='a1a2':
        prop=propose_public_a1(observations=h.unsqueeze(0),attention_mask=torch.ones((1,len(h)),dtype=torch.long),lens=lens,normalized_embeddings=emb)
        ids=prop.candidates[0,:,:256].to('cuda')
    else:ids=proposer.propose(h,fragments=method=='fragment512')
    sync();proposal_end=time.perf_counter()
    cache=new_context(prefix);tokens=[torch.tensor(128000,device='cuda')];scores=[];mses=[]
    for pos in range(1,len(h)):
        response=candidates(prefix,cache,ids[pos],pos).float()
        score=F.cosine_similarity(response,h[pos:pos+1],dim=-1);mse=(response-h[pos]).square().mean(-1)
        winner=ids[pos,score.argmax()];tokens.append(winner);scores.append(score);mses.append(mse)
        prefix.run_cached(winner.reshape(1,1),cache,pos)
    sync();verify_end=time.perf_counter()
    result={'tokens':torch.stack(tokens).cpu(),'candidates':ids.cpu(),'scores':torch.stack(scores).cpu(),'mse':torch.stack(mses).cpu()}
    sync();end=time.perf_counter()
    return result,{'total':end-start,'proposal_and_input':proposal_end-start,'verification_and_commit':verify_end-proposal_end,'output_transfer':end-verify_end}

def main():
    p=argparse.ArgumentParser();p.add_argument('--qualify',action='store_true');p.add_argument('--panel',default='fresh_r2');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=environment();sync();start=time.perf_counter();prefix=load_prefix();sync();load_seconds=time.perf_counter()-start
    tok=AutoTokenizer.from_pretrained(ASSETS/'backup',local_files_only=True)
    start=time.perf_counter();suffix_cache=build_suffix_cache(tok,len(prefix.embed_tokens.weight));suffix_seconds=time.perf_counter()-start
    cache_path=OUT/'tokenizer_suffix_cache.json'
    if not cache_path.exists():write(cache_path,suffix_cache)
    # Independent direct implementation checks batched tokenizer metadata construction.
    for v in [0,88,291,4147,13111,128000]+list(range(1000,1100)):
        assert suffix_cache[v]==suffix_ids(tok,v)
    batching={}
    if a.qualify:
        raw=intrinsic_table(prefix,256);sync();start=time.perf_counter();fast=intrinsic_table(prefix,4096);sync()
        batching={'fast_seconds':time.perf_counter()-start,'bitwise_equal':torch.equal(raw,fast),'different_values':int((raw!=fast).sum()),'max_abs_difference':float((raw.float()-fast.float()).abs().max())}
        native_chunk=4096 if batching['bitwise_equal'] else 256
        del raw,fast
    else:
        native_chunk=json.loads((X/'fragment_qualification.json').read_text())['native_chunk']
    build_times=[];stats=None
    for repeat in range(3):
        sync();start=time.perf_counter();proposer=PrefixFragmentProposer(prefix,suffix_cache);stats=proposer.build(native_chunk);sync();build_times.append(time.perf_counter()-start)
        if repeat<2:del proposer
    sync();start=time.perf_counter();lens,emb=prepare(prefix);sync();a1_prepare=time.perf_counter()-start;guard()
    setup={'environment':env,'prefix_load_seconds':load_seconds,'tokenizer_suffix_cache_seconds':suffix_seconds,'tokenizer_cache_sha256':digest(cache_path),'prefix_rebuild_seconds':build_times,'native_chunk':native_chunk,'cache':stats,'a1_setup_seconds':a1_prepare,'a1_prior_fit_cost':'not remeasured','prefix_sha256':digest(ASSETS/'backup/prefix.safetensors')}
    fixture=torch.tensor([[128000]+list(range(1000,1127))],device='cuda');h=prefix.forward_full(fixture)[0].cpu();largest={}
    for m in METHODS:
        result,phases=decode(prefix,h,m,proposer,lens,emb);guard();largest[m]={'phases':phases,'candidate_shape':list(result['candidates'].shape),'finite':bool(torch.isfinite(result['scores']).all())}
    new,_=decode(prefix,h[:16],'a1a2',proposer,lens,emb);old=original_comparator(prefix,h[:16],lens,emb)
    assert new['tokens'].tolist()==old['tokens']
    assert new['candidates'][1:].tolist()==[[v['token'] for v in r['checks']] for r in old['trace']]
    if a.qualify:
        # Full-record candidate identity check against the frozen development probe.
        opened=load_file(str(OUT/'fresh_r1/observations.safetensors'))['matched__book11_00']
        ids=proposer.propose(opened.float()).cpu()[1:]
        original=load_file(str(OUT/'fragment_dev_r1/matched__book11_00__union.safetensors'))['candidates']
        assert torch.equal(ids,original)
        write(X/'fragment_qualification.json',{'setup':setup,'native_chunk':native_chunk,'batching':batching,'largest':largest,'original_fragment_candidates_equal':True,'original_a1a2_candidates_tokens_equal':True,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        print('fragment qualification passed',batching,flush=True);return
    panel=OUT/a.panel;meta=json.loads((panel/'metadata.json').read_text());obs=load_file(str(panel/'observations.safetensors'))
    dst=OUT/(a.panel+'_predictions');dst.mkdir(exist_ok=False);entries=[]
    for i,row in enumerate(meta):
        for m in METHODS[i%3:]+METHODS[:i%3]:
            guard();first=None;phases=[]
            for repeat in range(3):
                result,timing=decode(prefix,obs[row['id']],m,proposer,lens,emb);phases.append(timing)
                if first is None:first=result
                elif not all(torch.equal(first[k],result[k]) for k in first):raise RuntimeError('nonreproducible result')
            start=time.perf_counter();path=dst/(row['id']+'__'+m+'.safetensors');save_file(first,str(path));sha=digest(path);io=time.perf_counter()-start
            times=[v['total'] for v in phases]
            entries.append({**row,'method':m,'path':str(path.relative_to(ROOT)),'sha256':sha,'seconds':times,'median_seconds':sorted(times)[1],'phases':phases,'io_and_hash_seconds':io,'logical_simulations':(len(first['tokens'])-1)*first['candidates'].shape[1]})
        print(row['id'],'all methods frozen',flush=True)
    sources=[Path(__file__),ROOT/'src/token_reconstruction/prefix_fragments.py',ROOT/'src/token_reconstruction/prefix_weight_metric.py',ROOT/'scripts/trr0014/native.py']
    write(X/(a.panel+'_prediction_receipt.json'),{'setup':setup,'entries':entries,'observations_sha256':digest(panel/'observations.safetensors'),'metadata_sha256':digest(panel/'metadata.json'),'implementation_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sources},'truth_read':False,'end_utc':environment()['utc'],'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    print('Complete240-cell prediction freeze',flush=True)
if __name__=='__main__':main()

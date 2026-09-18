"""No-truth prediction and timing for the frozen paired exploratory panel."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from token_reconstruction.component_crossover import propose_public_a1
from torch.nn import functional as F
from comparator import prepare,decode as original_comparator
import argparse,resource

@torch.no_grad()
def decode(prefix,h,method,metric,lens,emb):
    sync();start=time.perf_counter()
    h=h.to('cuda').float()
    if method=='metric64':
        ids=metric.propose(h,64)
    elif method=='raw64':
        ids=(F.normalize(h,dim=-1)@emb.T).topk(64,dim=-1).indices
    elif method=='a1a2':
        prop=propose_public_a1(observations=h.unsqueeze(0),attention_mask=torch.ones((1,len(h)),dtype=torch.long),lens=lens,normalized_embeddings=emb)
        ids=prop.candidates[0,:,:256]
    else:raise ValueError(method)
    cache=new_context(prefix);tokens=[torch.tensor(128000,device='cuda')];scores=[];residuals=[]
    for pos in range(1,len(h)):
        output=candidates(prefix,cache,ids[pos],pos).float()
        score=F.cosine_similarity(output,h[pos].unsqueeze(0),dim=-1)
        residual=(output-h[pos]).square().mean(-1)
        winner=ids[pos,score.argmax()]
        tokens.append(winner);scores.append(score);residuals.append(residual)
        prefix.run_cached(winner.reshape(1,1),cache,pos)
    result={'tokens':torch.stack(tokens).cpu().contiguous(),'candidates':ids.cpu().contiguous(),'scores':torch.stack(scores).cpu().contiguous(),'mse':torch.stack(residuals).cpu().contiguous()}
    sync()
    return result,time.perf_counter()-start

def setup():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=environment();t=time.perf_counter();prefix=load_prefix();sync();loadtime=time.perf_counter()-t
    # First-use and repeat rebuild costs both retained; no cached result silently treated free.
    rebuild=[];stats=None
    for _ in range(3):
        sync();t=time.perf_counter();metric=PrefixWeightMetric(prefix);stats=metric.build();sync();rebuild.append(time.perf_counter()-t)
        if len(rebuild)<3:del metric
    sync();t=time.perf_counter();lens,emb=prepare(prefix);sync();a1prep=time.perf_counter()-t
    guard()
    return prefix,metric,lens,emb,{'environment':env,'prefix_load_seconds':loadtime,'metric_rebuild_seconds':rebuild,'metric_cache':stats,'a1_asset_setup_seconds':a1prep,'a1_historical_fit_seconds':'not remeasured','prefix_sha256':digest(ASSETS/'backup/prefix.safetensors')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--qualify',action='store_true');p.add_argument('--panel',default='fresh_r1');args=p.parse_args()
    prefix,metric,lens,emb,setup_receipt=setup()
    fixture=torch.tensor([[128000]+list(range(1000,1127))],device='cuda')
    h=prefix.forward_full(fixture)[0].cpu()
    before=[p._version for p in prefix.parameters()]
    largest={}
    for m in ['metric64','raw64','a1a2']:
        result,seconds=decode(prefix,h,m,metric,lens,emb);largest[m]={'seconds':seconds,'valid':bool(torch.isfinite(result['scores']).all()),'shape':list(result['tokens'].shape)}
        guard()
    assert before==[p._version for p in prefix.parameters()]
    short=h[:16]
    new,_=decode(prefix,short,'a1a2',metric,lens,emb)
    old=original_comparator(prefix,short,lens,emb)
    assert new['tokens'].tolist()==old['tokens']
    assert new['candidates'][1:].tolist()==[[v['token'] for v in row['checks']] for row in old['trace']]
    if args.qualify:
        write(X/'confirmation_qualification.json',{'setup':setup_receipt,'largest':largest,'native_comparator_tokens_candidates_equal':True,'prefix_unchanged':True,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'qualified_positions':128,'qualified_candidates':256})
        print('qualification passed',flush=True);return
    assert (X/'confirmation_qualification.json').exists()
    panel=OUT/args.panel
    meta=json.loads((panel/'metadata.json').read_text())
    obs=load_file(str(panel/'observations.safetensors'))
    dst=OUT/(args.panel+'_predictions');dst.mkdir(exist_ok=False)
    receipt=X/(args.panel+'_prediction_receipt.json')
    methods=['metric64','raw64','a1a2'];entries=[]
    for i,row in enumerate(meta):
        guard();key=row['id']
        for m in methods[i%3:]+methods[:i%3]:
            times=[];first=None
            for repeat in range(3):
                result,elapsed=decode(prefix,obs[key],m,metric,lens,emb);times.append(elapsed)
                if first is None:first=result
                elif not all(torch.equal(first[k],result[k]) for k in first):raise RuntimeError('non-reproducible prediction')
            path=dst/(key+'__'+m+'.safetensors');save_file(first,str(path))
            entries.append({'id':key,'method':m,'group':row['group'],'condition':row['condition'],'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':times,'median_seconds':sorted(times)[1],'tokens':len(first['tokens'])})
        print(key,'frozen',flush=True)
    write(receipt,{'setup':setup_receipt,'entries':entries,'observations_sha256':digest(panel/'observations.safetensors'),'metadata_sha256':digest(panel/'metadata.json'),'implementation_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/token_reconstruction/prefix_weight_metric.py',ROOT/'scripts/trr0014/native.py']},'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'truth_read':False,'end_utc':environment()['utc']})
    print('complete prediction freeze',flush=True)
if __name__=='__main__':main()

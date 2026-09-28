"""Calibration and prediction-only component run; source truth never loaded."""
import sys,json,time,statistics,subprocess,hashlib,resource,argparse
from pathlib import Path
from engine import *
from shared import load_prefix,guard,environment,digest
from safetensors.torch import load_file
import transformers
R=Path(__file__).resolve().parents[2]; X=R/'experiments/agent4-joint-native-gate'; E=X/'evidence'
OLD=R.parent/'agent4-prefix-only-rescue'; ASSETS=OLD/'outputs/agent4-prefix-only-rescue'
def write(path,v):
    with path.open('x') as f:json.dump(v,f,indent=2)
def clocked(fn):
    sync();t=time.perf_counter();v=fn();sync();return v,time.perf_counter()-t

def context(prefix,rid,pos):
    t=time.perf_counter()
    old=json.loads((OLD/'experiments/agent4-prefix-only-rescue/evidence/final_discrete'/(rid+'.json')).read_text())
    tokens=old['tokens'][:pos]
    ex=CurrentToken(prefix)
    for v in tokens[1:]:ex.commit(v)
    sync();ctx_seconds=time.perf_counter()-t
    t=time.perf_counter();rng=torch.Generator().manual_seed(4401)
    draws=[int(torch.randint(len(prefix.embed_tokens.weight),(1,),generator=rng)) for _ in range(pos+2)]
    initial=draws[pos-1:pos+2]
    assert initial[0]==old['trace'][pos-1]['initial_token']
    return ex,initial,{'source_prediction_sha256':digest(OLD/'experiments/agent4-prefix-only-rescue/evidence/final_discrete'/(rid+'.json')),
        'committed_prefix_ids':tokens,'context_seconds':ctx_seconds,'initialization_seconds':time.perf_counter()-t,
        'prefix_length':pos,'initialization':'seed4401 native CPU random stream, no future predictions'}

def signature(ex):
    return [(i,k.data_ptr(),v.data_ptr(),k._version,v._version,k.shape[-2]) for i,(k,v) in ex.state.items()]

def qualify(prefix,obs):
    ex,initial,meta=context(prefix,'natural_84_1',18); win=Window(ex)
    before=signature(ex); cached={i:(k.clone(),v.clone()) for i,(k,v) in ex.state.items()}
    z=prefix.embed_tokens.weight[initial].float().detach().requires_grad_()
    h=obs['natural_84_1'][18:21].to('cuda').float()
    single=ex(z[0]).float(); window=win(z)
    assert torch.equal(single,window[0].float())
    gl=torch.autograd.grad(losses(single[None],h[:1]).sum(),z,retain_graph=True)[0]
    own=torch.autograd.grad(losses(window,h)[0],z,retain_graph=True)[0]
    assert torch.equal(gl,own)
    assert int(torch.count_nonzero(own[1:]))==0
    whole=torch.autograd.grad(losses(window,h).sum(),z,retain_graph=True)[0]
    parts=[torch.autograd.grad(losses(window,h)[j],z,retain_graph=j<2)[0] for j in range(3)]
    summed=torch.stack(parts).sum(0)
    relative=float((whole-summed).norm()/summed.norm().clamp_min(1e-12))
    assert relative<.02,relative  # BF16 derivative accumulation, not acceptance.
    ref,_,_=context(prefix,'natural_84_1',18)
    ys=[]
    for v in initial:
        with torch.no_grad():ys.append(ref(prefix.embed_tokens.weight[v]).float())
        ref.commit(v)
    assert torch.equal(torch.stack(ys),window.float())
    assert before==signature(ex)
    assert all(torch.equal(cached[i][0],k) and torch.equal(cached[i][1],v) for i,(k,v) in ex.state.items())
    assert prefix.rotary_emb.inv_freq.dtype==torch.float32
    assert not any(p.requires_grad or p.grad is not None for p in prefix.parameters())
    guard()
    return {'width1_output_exact':True,'width1_own_gradient_exact':True,'root_loss_future_gradient_zero':True,
        'window_output_matches_sequential_commits_exactly':True,'committed_cache_unchanged':True,
        'summed_vs_decomposed_gradient_relative_l2':relative,'gradient_check_tolerance':.02,
        'cross_observation_root_gradient_norm':float((whole[0]-own[0]).norm()),
        'rope_dtype':str(prefix.rotary_emb.inv_freq.dtype),'prefix_frozen':True,
        'largest_context_length':18,'provisional_width':3,'candidate_batch':1,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}

def calibrate(prefix,obs,vocab):
    ex=CurrentToken(prefix); win=Window(ex);rng=torch.Generator().manual_seed(4401)
    initial=[int(torch.randint(len(vocab.table),(1,),generator=rng)) for _ in range(3)]
    z=vocab.table[initial];h=obs['natural_11_0'][1:4].to('cuda').float()
    g=gradient(win,z,h,'joint')
    def fw(width):
        with torch.no_grad():return win(z[:width])
    jobs={'forward1':lambda:fw(1),'forward3':lambda:fw(3),
          'gradient_local1':lambda:gradient(win,z[:1],h[:1],'sequential'),
          'gradient_joint3':lambda:gradient(win,z,h,'joint'),
          'gradient_diagonal3':lambda:gradient(win,z,h,'diagonal'),
          'scan2':lambda:vocab.rank(z[0],g[0],2),'scan16':lambda:vocab.rank(z[0],g[0],16)}
    timings={}
    for key,fn in jobs.items():
        fn();sync(); timings[key]=[clocked(fn)[1] for _ in range(5)]
    rates={k:statistics.median(v) for k,v in timings.items()}
    cap=rates['forward1']+4*(rates['gradient_local1']+rates['scan16']+16*rates['forward1'])
    return {'common_context':'natural_11_0:1 with BOS-only cache','initial_ids':initial,
        'rates_seconds':rates,'samples_seconds':timings,'shared_work_cap_seconds':cap,
        'shared_wall_cap_seconds':2*cap+.01,'truth_read':False,
        'cap_definition':'1 forward1 +4*(gradient_local1+scan16+16*forward1)',
        'wall_definition':'2*calibrated_work_cap+0.01 seconds; no truth-based adjustment'}

p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','predict']);args=p.parse_args()
torch.set_num_threads(2);start=time.perf_counter()
prefix=load_prefix(torch.bfloat16,asset_root=ASSETS/'restore');guard();sync()
load_seconds=time.perf_counter()-start
obs,observation_load_seconds=clocked(lambda:load_file(str(ASSETS/'final/observations.safetensors')))
vocab,vocabulary_setup_seconds=clocked(lambda:Vocabulary(prefix.embed_tokens.weight.detach()))
common={'environment':environment(),'model_asset_root':str(ASSETS/'restore'),
        'model_sha256':digest(ASSETS/'restore/prefix.safetensors'),
        'observations_sha256':digest(ASSETS/'final/observations.safetensors'),
        'torch_path':torch.__file__,'transformers_path':transformers.__file__,
        'model_load_seconds':load_seconds,'observation_load_seconds':observation_load_seconds,
        'vocabulary_setup_seconds':vocabulary_setup_seconds,'radius':float(vocab.radius),
        'radius_rule':'0.75 median raw embedding L2 norm, from supplied prototype',
        'engine_sha256':digest(R/'scripts/agent4_joint/engine.py')}
if args.stage=='calibrate':
    result={**common,'qualification':qualify(prefix,obs),'calibration':calibrate(prefix,obs,vocab)}
    guard();result['post_import_seconds']=time.perf_counter()-start
    write(E/'calibration.json',result)
    print(json.dumps({'qualification':result['qualification'],'calibration':result['calibration']},indent=2))
else:
    calibration=json.loads((E/'calibration.json').read_text()); c=calibration['calibration']
    assert calibration['engine_sha256']==common['engine_sha256']
    assert calibration['model_sha256']==common['model_sha256']
    results=[]
    for rid,pos,modes in [('natural_11_1',9,['joint','diagonal','sequential']),
                          ('natural_84_1',18,['sequential','diagonal','joint'])]:
        guard();ex,initial,ctx=context(prefix,rid,pos);win=Window(ex)
        before=signature(ex)
        h,observation_transfer_seconds=clocked(lambda:obs[rid][pos:pos+3].to('cuda').float())
        # Untimed warm-up is qualification work and reported, not search credit.
        _,warmup_seconds=clocked(lambda:win(vocab.table[initial]))
        for mode in modes:
            result=search(win,vocab,h,initial,mode,c['rates_seconds'],c['shared_work_cap_seconds'],c['shared_wall_cap_seconds'])
            assert before==signature(ex)
            result.update(record=rid,position=pos,context=ctx,
                observation_transfer_seconds=observation_transfer_seconds,warmup_seconds=warmup_seconds)
            overhead=load_seconds+observation_load_seconds+vocabulary_setup_seconds+ctx['context_seconds']+ctx['initialization_seconds']+observation_transfer_seconds
            result['total_charged_seconds']=overhead+result['budget']['actual_search_seconds']
            path=E/(rid+'_'+str(pos)+'_'+mode+'.json');write(path,result)
            results.append({'path':str(path.relative_to(R)),'sha256':digest(path)})
            print(json.dumps({'record':rid,'position':pos,'mode':mode,'returned':result['returned_block'],
                'proposed_roots':len(result['root_proposals']),'verified_roots':len(result['root_verified']),
                'seconds':result['budget']['actual_search_seconds'],'stop':result['stop_reason']}),flush=True)
    guard();sync()
    write(E/'freeze.json',{**common,'calibration_sha256':digest(E/'calibration.json'),'predictions':results,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'post_import_seconds':time.perf_counter()-start,'truth_read':False,'record_scope':'two opened roots, no reconstruction panel'})

"""Bounded privileged diagnosis on already-opened rescue material only."""
import sys,json,time,hashlib,subprocess,functools
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts/agent4_rescue'))
from shared import load_prefix,guard,environment,digest
from executor import CurrentToken
import torch,transformers
from safetensors.torch import load_file
X=R/'experiments/agent4-holistic-reassessment';E=X/'evidence'
OLD=R/'experiments/agent4-prefix-only-rescue/evidence'
OUT=R/'outputs/agent4-prefix-only-rescue'
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)
torch.set_num_threads(2);start=time.perf_counter()
prefix=load_prefix(torch.bfloat16,asset_root=OUT/'restore');guard()
obs=load_file(str(OUT/'final/observations.safetensors'))
truth=json.loads((OUT/'final/evaluator_truth.json').read_text())
analysis=json.loads((E/'existing_trace_analysis.json').read_text())
source=(R/'scripts/agent4_rescue/search.py').read_text()
phases=defaultdict(float)
@contextmanager
def phase(name):
    torch.cuda.synchronize();t=time.perf_counter()
    yield
    torch.cuda.synchronize();phases[name]+=time.perf_counter()-t
def timed(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a,**kw):
            with phase(name):return fn(*a,**kw)
        return wrap
    return deco
profile=source.replace('        def verify(ids):',"        @timed('candidate_forwards_verification_bookkeeping')\n        def verify(ids):")
profile=profile.replace('        def rank(center):',"        @timed('vocabulary_ranking_and_shortlist')\n        def rank(center):")
for begin,end,name in [
    ('    table=prefix.embed_tokens.weight.detach()','    rng=', 'table_setup'),
    ('                z=table[best].float()', '                if not torch.isfinite(g)', 'input_forward_and_gradient')]:
    i=profile.index(begin);j=profile.index(end,i);indent=len(begin)-len(begin.lstrip())
    profile=profile[:i]+' '*indent+'with phase('+repr(name)+'):\n'+''.join('    '+l for l in profile[i:j].splitlines(True))+profile[j:]
profile=profile.replace('ex=CurrentToken(prefix)',"ex=timed('initial_cache_commit')(CurrentToken)(prefix)")
profile=profile.replace('tokens.append(best);ex.commit(best);sync()',"tokens.append(best);timed('token_cache_commit')(ex.commit)(best);sync()")
namespace={'phase':phase,'timed':timed};exec(compile(profile,'instrumented_discrete_replay','exec'),namespace)
profiles=[]
for rid in ['natural_11_0','natural_84_1']:
    phases.clear()
    replay=namespace['reconstruct'](prefix,obs[rid],'discrete',guard=guard)
    old=json.loads((OLD/'final_discrete'/(rid+'.json')).read_text())
    assert replay['tokens']==old['tokens']
    assert [[c['token'] for c in t['checks']] for t in replay['trace']]==[[c['token'] for c in t['checks']] for t in old['trace']]
    assert [[c['mse'] for c in t['checks']] for t in replay['trace']]==[[c['mse'] for c in t['checks']] for t in old['trace']]
    profiles.append({'record':rid,'seconds':replay['seconds'],'original_seconds':old['seconds'],'phases_seconds':dict(phases),'unattributed_seconds':replay['seconds']-sum(phases.values()),'all_candidate_orders_values_and_tokens_identical':True,'per_position_seconds':[t['seconds'] for t in replay['trace']]})
# This is intentionally privileged. Skip earlier search, install truth-only diagnostic
# context and advance the original random stream to exactly the same position.
privileged=source.replace('guard=None):','guard=None,diagnostic_prefix=None):',1)
privileged=privileged.replace('tokens=[128000];traces=[];ex=CurrentToken(prefix)',"""tokens=[128000];traces=[];ex=CurrentToken(prefix)
    for token in diagnostic_prefix[1:]:
        tokens.append(token);ex.commit(token)
        torch.randint(len(table),(1,),generator=rng)""")
privileged=privileged.replace('for pos in range(1,len(observation)):','for pos in range(len(tokens),len(observation)):')
space={};exec(compile(privileged,'privileged_correct_prefix_only','exec'),space)
checks=[]
for error in analysis['errors']:
    rid=error['record'];pos=error['position'];gold=truth[rid];target=obs[rid][pos].to('cuda').float()
    result=space['reconstruct'](prefix,obs[rid][:pos+1],'discrete',diagnostic_prefix=gold[:pos])
    trace=result['trace'][0];found=any(c['token']==gold[pos] for c in trace['checks'])
    ex=CurrentToken(prefix)
    for token in gold[1:pos]:ex.commit(token)
    with torch.no_grad():
        true_y=ex(prefix.embed_tokens.weight[gold[pos]]).float()
        forced_mse=float((true_y-target).square().mean())
        forced_accept=torch.allclose(true_y,target,atol=1e-5,rtol=1e-5)
        chosen_y=ex(prefix.embed_tokens.weight[trace['token']]).float()
    old=json.loads((OLD/'final_discrete'/(rid+'.json')).read_text())['trace'][pos-1]
    order_same=[c['token'] for c in old['checks']]==[c['token'] for c in trace['checks']]
    if error['first_error']:assert order_same
    checks.append({'record':rid,'position':pos,'first_error':error['first_error'],'original_true_proposed':error['true_proposed'],'correct_prefix_true_proposed':found,'correct_prefix_selected_true':trace['token']==gold[pos],'correct_prefix_selected_mse':trace['best_mse'],'forced_genuine_cached_mse':forced_mse,'forced_genuine_passes_unchanged_tolerance':forced_accept,'original_candidate_order_reproduced':order_same,'checks':len(trace['checks']),'gradients':trace['gradient_steps'],'seconds':result['seconds']})
    write(E/(rid+'_'+str(pos)+'_privileged_trace.json'),result)
guard();torch.cuda.synchronize()
write(E/'bounded_diagnostics.json',{'environment':environment(),'loaded_assets':str(OUT/'restore'),'torch_path':torch.__file__,'transformers_path':transformers.__file__,'sys_path':sys.path,'source_sha256':digest(R/'scripts/agent4_rescue/search.py'),'profiles':profiles,'correct_prefix_checks':checks,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'post_import_total_seconds':time.perf_counter()-start,'scope':'Two opened-record instrumented replays plus seven privileged correct-prefix/forced-candidate checks. Not new reconstruction performance, no threshold or algorithm changed.'})
print(json.dumps({'profiles':profiles,'correct_prefix_checks':checks},indent=2))

"""Truth-free adapters over validated public proposal, decode and timing interfaces."""
from __future__ import annotations
import contextlib,gc,importlib,json,os,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'scripts'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from scripts.agent3_shortlists.core import binding,verify,sha,rank_scores
from scripts.agent3_shortlists.predict import load_models,b1_logits,STATE_SHA,READOUT_SHA
from token_reconstruction import a1a2_configuration_search as native
from token_reconstruction.component_crossover import propose_public_a1
import trr0004_predict_confirmation as legacy
import trr0004_fresh_confirmation as timing

METHODS=('a1_k256','b1_k256','b1_k16','a1_k16','b1_alone')
ASSETS=ROOT.parent/'agent3-b1-small-budget-a2/outputs/agent3-b1-small-budget-a2/assets'
SNAPSHOT=Path('/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6')
CODE_SHA={
 'scripts/trr0004_predict_confirmation.py':'36f6aa7b4493c60b257b3896c975f523595da912e14a97dba3f6440d419e8427',
 'src/token_reconstruction/a1a2_configuration_search.py':'608bead6291353cacaa700a85f3dd619fb3261375e6fd5ba19cbc43fd738315b',
 'src/token_reconstruction/component_crossover.py':'2cd03fda1f29f30d0a92246252ba2d2d89d3890abc6739ae0c8bae0a5c105b7d',
 'scripts/trr0004_fresh_confirmation.py':'482c3a75d2fc641f90fe9c531584e2be14aefe914d97b246fe3633bdaa3d2945',
 'reference/strict_bos/round001_teacher.py':'10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd'}

def sync():torch.cuda.synchronize()

def configure():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def resources():
    free,total=torch.cuda.mem_get_info()
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return {'gpu_free_bytes':free,'gpu_total_bytes':total,'gpu_allocated_bytes':torch.cuda.memory_allocated(),'gpu_reserved_bytes':torch.cuda.memory_reserved(),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'host_available_bytes':available}

def guard():
    r=resources()
    if r['gpu_free_bytes']<3*2**30 or r['gpu_reserved_bytes']>8*2**30 or r['rss_peak_bytes']>12*2**30 or r['host_available_bytes']<6*2**30:raise RuntimeError('Resource guard failed: '+json.dumps(r))
    return r

def policy(k):
    return native.ResolvedPolicy(native.PolicySpec(kind='fixed',score_rule='direct_cosine',schedule=(k,),fast_path_id='off',fast_path_threshold=None,routing_signal=None,gate_mode=None,terminal_action='commit_last_winner'),())

class Resources:
    def __init__(self):
        start=time.perf_counter();self.device=torch.device('cuda');guard()
        for p,s in CODE_SHA.items():
            if sha(ROOT/p)!=s:raise ValueError('native implementation hash changed: '+p)
        self.package,self.b1,readout,_=load_models(ASSETS/'package',ASSETS/'public_a1_lens.pt',ROOT/'reference/strict_bos/round001_teacher.py',self.device)
        self.prefix,self.lens,self.readout,self.native_evidence=legacy._load_public_prefix(snapshot=SNAPSHOT,reference_path=ROOT/'reference/strict_bos/round001_teacher.py',lens_path=ASSETS/'public_a1_lens.pt',embedding_path=ASSETS/'package/readout/public_normalized_embeddings.safetensors',device=self.device)
        if not torch.equal(readout,self.readout):raise ValueError('B1 and A1 readouts differ')
        del readout;gc.collect();torch.cuda.empty_cache();sync()
        self.load_seconds=time.perf_counter()-start;self.resource_after_load=guard()

class Trace:
    """Time native calls and retain decisions without altering their tensors."""
    def __init__(self,prefix):
        self.prefix=prefix;self.phases={'candidate_branch_simulation_seconds':0.,'direct_scoring_seconds':0.,'cache_commit_seconds':0.};self.scores=[];self.commits=[];self.candidate_counts=[];self.new_caches=0
    @contextlib.contextmanager
    def enabled(self):
        candidate=native._candidate_hidden;score=native.score_candidates;run=self.prefix.run_cached;new=self.prefix.new_cache
        def candidates(*args,**kwargs):
            sync();t=time.perf_counter();result=candidate(*args,**kwargs);sync();self.phases['candidate_branch_simulation_seconds']+=time.perf_counter()-t
            self.candidate_counts.append(int(kwargs['candidate_ids'].numel()));return result
        def scores(*args,**kwargs):
            sync();t=time.perf_counter();result=score(*args,**kwargs);sync();self.phases['direct_scoring_seconds']+=time.perf_counter()-t
            self.scores.append(result.detach().clone());return result
        def cached(ids,cache,start_pos):
            # Batch one is the method's own committed cache; simulated candidates have K16/256 rows.
            if ids.shape[0]!=1:return run(ids,cache,start_pos)
            sync();t=time.perf_counter();result=run(ids,cache,start_pos);sync();self.phases['cache_commit_seconds']+=time.perf_counter()-t
            self.commits.append((start_pos,ids.detach().clone()));return result
        def new_cache():self.new_caches+=1;return new()
        native._candidate_hidden=candidates;native.score_candidates=scores
        self.prefix.run_cached=cached;self.prefix.new_cache=new_cache
        try:yield self
        finally:
            native._candidate_hidden=candidate;native.score_candidates=score
            self.prefix.run_cached=run;self.prefix.new_cache=new

class Adapter:
    def __init__(self,resources,method,instrument=True):
        if method not in METHODS:raise ValueError(method)
        self.r=resources;self.method=method;self.k=1 if method=='b1_alone' else int(method.split('_k')[1]);self.instrument=instrument;self.calls=[];self.outputs=[];self.last=None
    @torch.inference_mode()
    def __call__(self,h,mask,pos):
        guard();sync();begin=time.perf_counter()
        observations=h.detach().cpu().unsqueeze(0);m=mask.detach().cpu().long().unsqueeze(0);p=pos.detach().cpu().long().unsqueeze(0)
        proposal_start=time.perf_counter()
        if self.method.startswith('a1'):
            proposal=propose_public_a1(observations=observations,attention_mask=m,lens=self.r.lens,normalized_embeddings=self.r.readout,max_k=512,chunk=256)
            candidates=proposal.candidates[:,:,:self.k].contiguous();scores=proposal.scores[:,:,:self.k].contiguous();confidence=proposal.top1_confidence
        else:
            logits=b1_logits(self.r.b1,self.r.readout,h,mask,pos)
            if self.k==1:
                ids=logits.argmax(-1,keepdim=True);values=logits.gather(-1,ids)
            else:ids,values=rank_scores(logits,256)
            candidates=torch.full((1,128,self.k),-1,dtype=torch.int64);candidates[:,1:]=ids[:,:self.k].cpu()
            scores=torch.zeros((1,128,self.k));scores[:,1:]=values[:,:self.k].cpu()
            confidence=torch.zeros((1,128));del logits,ids,values
        sync();proposal_seconds=time.perf_counter()-proposal_start
        trace=Trace(self.r.prefix);decode_start=time.perf_counter()
        if self.k==1:
            predictions=torch.cat([torch.tensor([[128000]]),candidates[:,:,0][:,1:]],dim=1)
            simulations=commits=0;a2_scores=torch.empty((127,0));commit_ids=torch.empty((0,),dtype=torch.int64)
        else:
            with trace.enabled() if self.instrument else contextlib.nullcontext():
                decoded=native.decode_policy(observations=observations,attention_mask=m,position_ids=p,candidates=candidates,a1_confidence=confidence,precut=self.r.prefix,device=self.r.device,policy=policy(self.k),record_batch_size=1)
            predictions=decoded.predictions;simulations=decoded.executed_candidate_simulations;commits=decoded.prefix_commit_tokens
            if simulations!=127*self.k or commits!=128:raise ValueError('native work counts changed')
            if self.instrument:
                a2_scores=torch.cat(trace.scores).cpu();commit_ids=torch.cat([x[1].flatten().cpu() for x in trace.commits])
                if trace.new_caches!=1 or [x[0] for x in trace.commits]!=list(range(128)) or not torch.equal(commit_ids,predictions[0]):raise ValueError('cache does not match own reconstructed tokens')
                if sum(trace.candidate_counts)!=simulations:raise ValueError('candidate count mismatch')
            else:a2_scores=torch.empty((127,0));commit_ids=torch.empty((0,),dtype=torch.int64)
        sync();decode_seconds=time.perf_counter()-decode_start
        if not torch.all(candidates[:,1:].eq(predictions[:,1:,None]).any(-1)):raise ValueError('winner outside fixed candidate list')
        self.last={'candidates':candidates[0,1:].clone(),'proposal_scores':scores[0,1:].clone(),'predictions':predictions[0].clone(),'a2_scores':a2_scores,'committed_tokens':commit_ids}
        self.outputs.append(self.last)
        self.calls.append({'proposal_seconds':proposal_seconds,'decode_seconds':decode_seconds,**trace.phases,'adapter_total_seconds':time.perf_counter()-begin,'candidate_simulations':simulations,'prefix_commit_tokens':commits,'new_cache_count':trace.new_caches,'instrumented':self.instrument})
        return predictions[0].to(h.device)


def warmed(resources,method,h,m,p):
    a=Adapter(resources,method)
    # The validated interface retains first measured output and checks all later repeats.
    prediction,times=timing.run_warmed_prediction(observations=h,attention_mask=m,position_ids=p,predictor=a,device='cuda',warmup_runs=1,measured_runs=3)
    if any(not r['repeated_prediction_exact'] for r in times['records']):raise ValueError('repeat prediction mismatch')
    # Only one record per call allows round-robin arm scheduling without changing native record geometry.
    if len(h)!=1:raise ValueError('one record per timed phase required')
    for later in a.outputs[2:]:
        if any(not torch.equal(later[k],a.outputs[1][k]) for k in a.outputs[1]):raise ValueError('repeat candidate/score/commit mismatch')
    if not torch.equal(a.outputs[1]['predictions'],prediction[0]):raise ValueError('trace differs from designated prediction')
    return prediction,a.outputs[1],times,a.calls

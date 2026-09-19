"""Read the unchanged fitted soft embedding directly against every vocabulary row."""
from types import MethodType
import time
import torch
from torch.nn import functional as F
from discrete_soft import DiscreteSoftVocabulary
from fast_soft import BF16Mixture
from reuse_stream import ensure_owned_stream
CONFIGS=[('warm'+str(steps),steps) for steps in [32,64,128,256]]
READOUTS=['raw_l2','raw_cosine','white_l2','white_cosine']

class MeanReadout:
    def __init__(self,prefix,steps):
        self.soft=DiscreteSoftVocabulary(prefix,'gini003',.003,0.,32);self.soft.steps=steps;self.steps=steps
        self.soft.capture_stream=torch.cuda.Stream();self.soft.ensure=MethodType(ensure_owned_stream,self.soft)
        self.weight=self.soft.weight.float();self.r=self.soft.transform
        self.raw_norm=self.weight.square().sum(-1)
        self.white_norm=torch.empty_like(self.raw_norm)
        for start in range(0,len(self.weight),4096):
            self.white_norm[start:start+4096]=(self.weight[start:start+4096]@self.r).square().sum(-1)
    @property
    def capture_events(self):return self.soft.capture_events
    @torch.no_grad()
    def score_tables(self,mean):
        raw=mean@self.weight.T;white=((mean@self.r)@self.r.T)@self.weight.T
        return {'raw_l2':2*raw-self.raw_norm[None],'raw_cosine':raw/self.raw_norm.clamp_min(1e-24).sqrt()[None],
                'white_l2':2*white-self.white_norm[None],'white_cosine':white/self.white_norm.clamp_min(1e-24).sqrt()[None]}
    def decode(self,h):
        torch.cuda.synchronize();start=time.perf_counter()
        old,stats=self.soft.decode(h);torch.cuda.synchronize();readout_start=time.perf_counter()
        with torch.no_grad():
            probability=F.softmax(self.soft.logits[:len(h)-1],dim=-1)
            mean=BF16Mixture.apply(probability,self.soft.weight)
            tables=self.score_tables(mean);out={'warm_'+k:v for k,v in old.items()}
            for name,score in tables.items():
                ids=torch.cat([torch.full((1,),128000,device='cuda',dtype=torch.long),score.argmax(-1)])
                out[name]=ids.cpu()
            out['fitted_embedding']=mean.cpu();out['final_confidence']=self.soft.confidence[:len(h)-1].cpu()
            finite=all(bool(torch.isfinite(v).all()) for v in tables.values())
        torch.cuda.synchronize();end=time.perf_counter()
        return out,{'total_seconds':end-start,'soft':stats,'readout_seconds_all_four':end-readout_start,
          'whole_sequence_prefix_forwards':self.steps+1,'whole_sequence_prefix_backwards':self.steps,
          'readout_vocabulary_products':2,'additional_mixture_products':1,'readout_numeric_valid':finite,
          'readout_numerics':'FP32_TF32_DISABLED; BF16 mixture operands withFP32accumulation',
          'vocabulary_entries_per_sweep':len(self.weight),'shortlist_size':None,'separate_candidate_verification_calls':0,
          'model_parameter_updates':0,'timing_scope':'unchanged soft decode plus all four readouts and saved fitted embedding; not single-readout timing'}

@torch.no_grad()
def score_reference(engine,mean):
    mean=mean.to('cuda');tables=engine.score_tables(mean)
    pos=torch.tensor([0,len(mean)//2,len(mean)-1],device='cuda');tok=torch.tensor([0,7,len(engine.weight)//2,len(engine.weight)-1],device='cuda')
    e=engine.weight[tok].cpu().double();q=mean[pos].cpu().double();r=engine.r.cpu().double()
    exact={}
    for name,transform in [('raw',torch.eye(q.shape[1],dtype=torch.float64)),('white',r)]:
        a=q@transform;b=e@transform;dot=a@b.T;norm=b.square().sum(-1)
        exact[name+'_l2']=2*dot-norm[None];exact[name+'_cosine']=dot/norm.clamp_min(1e-24).sqrt()[None]
    checks=[]
    for name,reference in exact.items():
        actual=tables[name][pos[:,None],tok[None]].cpu().double()
        checks.append({'readout':name,'actual':actual.tolist(),'float64_reference':reference.tolist(),
          'maximum_absolute_error':float((actual-reference).abs().max()),'all_close':bool(torch.allclose(actual,reference,rtol=2e-4,atol=1e-5))})
    return {'checks':checks,'all_close':all(c['all_close'] for c in checks),'rtol':2e-4,'atol':1e-5}

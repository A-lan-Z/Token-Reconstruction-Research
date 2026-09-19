"""Prototype: cached directional curvature, with full-vocabulary FP32 scoring."""
from types import MethodType
import time
import torch
from torch.nn import functional as F
from discrete_soft import DiscreteSoftVocabulary
from discrete_parallel import DiscreteParallel
from warm_refinement import WarmProximal,WarmRefinement
from reuse_stream import ensure_owned_stream

CONFIGS=[(f"warm{warm}_{metric}_a{str(alpha).replace('.','p')}_c{str(scale).replace('.','p')}",warm,metric,alpha,scale)
         for warm in [64,128] for metric in ["raw","white"]
         for alpha,scale in [(.1,1.),(.1,3.),(.5,1.),(.5,3.),(1.,1.)]]

class CachedProximal(WarmProximal):
    def __init__(self,soft,metric,alpha,scale):
        super().__init__(soft,metric,scale)
        self.alpha=alpha;self.scale=scale;self.samples=8
        self.score_weight=self.weight.float()
        self.probes=torch.zeros((8,127,2048),device="cuda")
        self.curvature=torch.ones(127,device="cuda")
        self.quadratic_norm=torch.zeros((127,len(self.weight)),device="cuda")
        self.capture_stream=torch.cuda.Stream()
        self.ensure=MethodType(ensure_owned_stream,self)
    def collect(self,initial):
        length=len(initial);pe,mask=self.geometry[length]
        z=self.weight[initial.to("cuda")[1:]].float().detach().requires_grad_()
        predicted=DiscreteParallel.diagonal_forward(self,torch.cat([self.bos,z[None]],dim=1),pe,mask)
        normalized=F.normalize(predicted[:,1:],dim=-1)
        generator=torch.Generator().manual_seed(200042)
        signs=(torch.randint(0,2,(8,1,127,2048),generator=generator)*2-1).to(device="cuda",dtype=torch.float32)
        samples=[];estimates=[]
        for k in range(8):
            g=torch.autograd.grad((normalized*signs[k,:,:length-1]).sum(),z,retain_graph=k<7)[0]
            samples.append(g)
            dual=g@self.rinv.T if self.kind=="white" else g
            estimates.append(dual.square().mean(-1))
        probes=torch.stack(samples).detach();curvature=torch.stack(estimates).mean(0).detach()
        # Return diagnostic values before the caller decides whether qualification passes.
        with torch.no_grad():
            self.probes[:,:length-1].copy_(probes);self.curvature[:length-1].copy_(curvature)
        return probes,curvature
    @torch.no_grad()
    def build_quadratic(self,length):
        count=length-1
        self.quadratic_norm[:count].copy_(self.alpha*self.curvature[:count,None]*self.norm[None])
        if self.alpha<1:
            for probe in self.probes[:,:count]:
                projection=probe@self.score_weight.T
                self.quadratic_norm[:count].add_((1-self.alpha)*projection.square()/self.samples)
    def train_step(self,length):
        self.gradient.zero_();self.evaluate(length).backward()
        with torch.no_grad():
            count=length-1;z=self.z[:count];g=self.gradient[:count]
            metric_z=(z@self.r)@self.r.T if self.kind=="white" else z
            bz=self.alpha*self.curvature[:count,None]*metric_z
            if self.alpha<1:
                probes=self.probes[:,:count]
                bz=bz+(1-self.alpha)*((probes*z[None]).sum(-1)[...,None]*probes).mean(0)
            scores=(self.scale*bz-g)@self.score_weight.T-.5*self.scale*self.quadratic_norm[:count]
            ids=scores.argmax(-1)
            self.tokens[1:length].copy_(ids);self.z[:count].copy_(self.weight[ids].float())
    def decode(self,h,initial):
        torch.cuda.synchronize();start=time.perf_counter();length=len(h)
        capture=self.ensure(length);self.reset(h,initial)
        torch.cuda.synchronize();probe_start=time.perf_counter()
        probes,curvature=self.collect(initial)
        torch.cuda.synchronize();cache_start=time.perf_counter()
        self.build_quadratic(length)
        torch.cuda.synchronize();cache_end=time.perf_counter()
        out,stats=super().decode(h,initial)
        # This scalar gate concerns numerical validity only, never hidden-token correctness.
        finite=bool(torch.isfinite(curvature).all() and torch.isfinite(self.quadratic_norm[:length-1]).all())
        positive=bool((curvature>0).all())
        stats.update(sensitivity_seconds=cache_start-probe_start,quadratic_cache_seconds=cache_end-cache_start,
          extra_initialization_seconds=probe_start-start-capture,
          curvature=curvature.cpu().tolist(),curvature_seed=200042,
          curvature_shrinkage=self.alpha,curvature_multiplier=self.scale,
          curvature_numeric_valid=finite and positive,
          score_numerics="FP32_TF32_DISABLED",score_embedding_cache_bytes=self.score_weight.numel()*4,
          quadratic_cache_bytes=self.quadratic_norm.numel()*4,curvature_fixed_during_refinement=True,
          sensitivity_vector_jacobian_products=8,sensitivity_forward_evaluations=1,
          quadratic_cache_vocabulary_products=8 if self.alpha<1 else 0,
          whole_sequence_prefix_forwards=34,whole_sequence_prefix_backwards=40,
          full_vocabulary_sweeps=32+(8 if self.alpha<1 else 0),capture_seconds=capture+stats["capture_seconds"])
        torch.cuda.synchronize();stats["total_seconds"]=time.perf_counter()-start
        return out,stats

class CachedRefinement(WarmRefinement):
    def __init__(self,prefix,warm,metric,alpha,scale):
        self.soft=DiscreteSoftVocabulary(prefix,"gini003",.003,0.,32);self.soft.steps=warm
        self.soft.capture_stream=torch.cuda.Stream();self.soft.ensure=MethodType(ensure_owned_stream,self.soft)
        self.direct=CachedProximal(self.soft,metric,alpha,scale);self.warm=warm
        assert self.soft.metric is self.direct.metric and self.soft.weight is self.direct.weight
    def decode(self,h):
        out,stats=super().decode(h)
        stats["whole_sequence_prefix_forwards"]+=1;stats["whole_sequence_prefix_backwards"]+=8
        stats["full_vocabulary_sweeps"]+=stats["direct"]["quadratic_cache_vocabulary_products"]
        stats["curvature_numeric_valid"]=stats["direct"]["curvature_numeric_valid"]
        return out,stats

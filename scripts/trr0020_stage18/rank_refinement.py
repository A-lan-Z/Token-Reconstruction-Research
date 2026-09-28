"""Direct full-vocabulary local loss with a current-residual rank-one term."""
from types import MethodType
import torch
from discrete_soft import DiscreteSoftVocabulary
from warm_refinement import WarmProximal,WarmRefinement
from reuse_stream import ensure_owned_stream

CONFIGS=[(f"warm{warm}_{metric}_b{str(beta).replace('.','p')}_g{str(gamma).replace('.','p')}",warm,metric,beta,gamma)
         for warm in [64,128] for metric in ['raw','white'] for beta in [.01,.03,.1] for gamma in [0.,1.]]

class RankProximal(WarmProximal):
    def __init__(self,soft,metric,beta,gamma):
        super().__init__(soft,metric,beta)
        self.gamma=gamma;self.score_weight=self.weight.float()
        self.numeric_valid=torch.ones((),device='cuda',dtype=torch.bool)
        self.capture_stream=torch.cuda.Stream();self.ensure=MethodType(ensure_owned_stream,self)
    @torch.no_grad()
    def reset(self,h,initial=None):
        super().reset(h,initial);self.numeric_valid.fill_(True)
    @torch.no_grad()
    def vocabulary_scores(self,z,g,error):
        loss=error.clamp_min(1e-8)
        if self.kind=='white':
            dual=g@self.rinv.T;metric_z=(z@self.r)@self.r.T
        else:dual=g;metric_z=z
        lam=(self.beta*dual.square().sum(-1)/(2*loss)).clamp(1e-6,1e6)
        linear=g@self.score_weight.T-(g*z).sum(-1,keepdim=True)
        distance=(self.norm[None]+(z@self.r).square().sum(-1,keepdim=True)-2*(metric_z@self.score_weight.T)).clamp_min(0)
        scores=-linear-.5*lam[:,None]*distance
        if self.gamma:scores=scores-self.gamma*linear.square()/(4*loss[:,None])
        return scores
    def train_step(self,length):
        self.gradient.zero_();self.evaluate(length).backward()
        with torch.no_grad():
            count=length-1
            scores=self.vocabulary_scores(self.z[:count],self.gradient[:count],self.error[:count])
            self.numeric_valid.logical_and_(torch.isfinite(scores).all())
            ids=scores.argmax(-1)
            self.tokens[1:length].copy_(ids);self.z[:count].copy_(self.weight[ids].float())
    def decode(self,h,initial):
        out,stats=super().decode(h,initial)
        stats.update(score_numeric_valid=bool(self.numeric_valid),rank_strength=self.gamma,
          rank_curvature='gamma*g*g.T/(2*max(observed_cosine_error,1e-8)); updated each iteration',
          isotropic_curvature='beta*dual_gradient_norm_squared/(2*max(error,1e-8)); clamped1e-6..1e6',
          full_vocabulary_sweeps=64,vocabulary_products_per_update=2,score_numerics='FP32_TF32_DISABLED',
          score_embedding_cache_bytes=self.score_weight.numel()*4,extra_sensitivity_calls=0,
          quadratic_cache_bytes=0,curvature_fixed_during_refinement=False,
          decision='direct argmax over all128256 tokens of current rank-one local quadratic')
        return out,stats

class RankRefinement(WarmRefinement):
    def __init__(self,prefix,warm,metric,beta,gamma):
        self.soft=DiscreteSoftVocabulary(prefix,'gini003',.003,0.,32);self.soft.steps=warm
        self.soft.capture_stream=torch.cuda.Stream();self.soft.ensure=MethodType(ensure_owned_stream,self.soft)
        self.direct=RankProximal(self.soft,metric,beta,gamma);self.warm=warm
        assert self.soft.metric is self.direct.metric and self.soft.weight is self.direct.weight
    def decode(self,h):
        out,stats=super().decode(h)
        stats['full_vocabulary_sweeps']+=32
        stats['score_numeric_valid']=stats['direct']['score_numeric_valid']
        return out,stats

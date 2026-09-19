"""Full-vocabulary warm reconstruction with fixed normalized-prefix sensitivity."""
from types import MethodType
import time
import torch
from discrete_soft import DiscreteSoftVocabulary
from warm_refinement import WarmProximal,WarmRefinement
from reuse_stream import ensure_owned_stream
from sensitivity import estimate_curvature,sensitivity_step

CONFIGS=[(f"warm{warm}_{metric}_gn{str(multiplier).replace('.','p')}",warm,metric,multiplier)
         for warm in [64,128] for metric in ["raw","white"] for multiplier in [.1,.3,1.]]

class SensitivityProximal(WarmProximal):
    train_step=sensitivity_step
    def __init__(self,soft,metric,multiplier):
        super().__init__(soft,metric,multiplier)
        self.multiplier=multiplier
        self.curvature=torch.ones(127,device="cuda")
        self.capture_stream=torch.cuda.Stream()
        self.ensure=MethodType(ensure_owned_stream,self)
    def decode(self,h,initial):
        torch.cuda.synchronize();start=time.perf_counter()
        length=len(h);capture=self.ensure(length);self.reset(h,initial)
        torch.cuda.synchronize();estimate_start=time.perf_counter()
        estimated=estimate_curvature(self,initial,8,200042)
        with torch.no_grad():self.curvature[:length-1].copy_(estimated)
        torch.cuda.synchronize();estimate_end=time.perf_counter()
        output,stats=super().decode(h,initial)
        torch.cuda.synchronize();end=time.perf_counter()
        stats.update(total_seconds=end-start,capture_seconds=capture+stats["capture_seconds"],
          sensitivity_seconds=estimate_end-estimate_start,
          extra_initialization_seconds=estimate_start-start-capture,
          whole_sequence_prefix_forwards=34,whole_sequence_prefix_backwards=40,
          sensitivity_forward_evaluations=1,sensitivity_vector_jacobian_products=8,
          curvature_estimator="eight fixed Rademacher probes of normalized prefix; trace/dimension",
          curvature_seed=200042,curvature=estimated.cpu().tolist(),curvature_multiplier=self.multiplier,
          curvature_recomputed_each_input=True,curvature_fixed_during_refinement=True)
        return output,stats

class SensitivityRefinement(WarmRefinement):
    def __init__(self,prefix,warm,metric,multiplier):
        self.soft=DiscreteSoftVocabulary(prefix,"gini003",.003,0.,32);self.soft.steps=warm
        self.soft.capture_stream=torch.cuda.Stream();self.soft.ensure=MethodType(ensure_owned_stream,self.soft)
        self.direct=SensitivityProximal(self.soft,metric,multiplier);self.warm=warm
        assert self.soft.metric is self.direct.metric and self.soft.weight is self.direct.weight
    def decode(self,h):
        output,stats=super().decode(h)
        stats["whole_sequence_prefix_forwards"]+=1;stats["whole_sequence_prefix_backwards"]+=8
        return output,stats

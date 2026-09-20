"""Full-vocabulary optimizer with a byte-qualified pointwise fusion variant."""
import torch
from first_optimizer import FirstOptimizer
from exact_step import update
class ExactOptimizer(FirstOptimizer):
    def __init__(self,prefix,exact):
        self.exact=exact
        super().__init__(prefix,False)
    def train_step(self,length):
        if self.stage=="warm" or not self.exact:return super().train_step(length)
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        with torch.no_grad():
            new,info=update(self.logits[:length-1],self.gradient[:length-1],self.position_error[:length-1],2.,4)
            self.logits[:length-1].copy_(new)
            record=torch.stack([info["requested_budget"].mean(),info["formula_kl"].mean(),info["budget_used"].mean(),
              info["normalized_step"].amax(),info["active"].float().mean()])
            self.mirror_update_trace.index_copy_(0,self.counter-1,record[None])
    def decode(self,h,steps=64,replay=True):
        out,stats=super().decode(h,steps,replay)
        stats["update_implementation"]="exact_pointwise_fp32" if self.exact else "original_torch"
        return out,stats

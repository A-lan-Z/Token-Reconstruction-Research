"""Observed-error scaling for the qualified full-vocabulary mirror optimizer."""
import torch
from mirror_optimizer import MirrorOptimizer as Base,CHECKPOINTS,AUX_KEYS,gradient_reference
from adaptive_step import update
CONFIGS=[(f"warm{warm}_tau{tau}_error",warm,tau) for warm in [0,64] for tau in [.01,.1,1.]]
class MirrorOptimizer(Base):
    def train_step(self,length):
        if self.stage=="warm":return super().train_step(length)
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        with torch.no_grad():
            new,info=update(self.logits[:length-1],self.gradient[:length-1],self.tau,self.position_error[:length-1])
            self.logits[:length-1].copy_(new)
            record=torch.stack([info["rate"].mean(),info["weighted_variance"].mean(),
              info["gradient_span"].mean(),info["update_span"].mean(),info["update_span"].amax()])
            self.mirror_update_trace.index_copy_(0,self.counter-1,record[None])
    def decode(self,h,replay=True):
        output,stats=super().decode(h,replay=replay)
        stats.update(error_scaling="sqrt(clamp(current observed cosine error / .05,0,1)); no positive floor",
          numerical_identity="dev30_error_scaled_full_vocab_mirror; no change to warm stage or full-vocabulary readout")
        return output,stats

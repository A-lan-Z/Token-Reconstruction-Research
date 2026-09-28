"""Delayed higher-power observed-error objective; no additional prefix passes."""
from types import MethodType
import torch
from torch.nn import functional as F
from discrete_soft import DiscreteSoftVocabulary,straight_through_mix
from fast_soft import BF16Mixture
from reuse_stream import ensure_owned_stream
CONFIGS=[(f"steps{steps}_power{power}",steps,power) for steps in [64,128] for power in [1,2,4]]
class FocusSoft(DiscreteSoftVocabulary):
    def __init__(self,prefix,steps,power):
        super().__init__(prefix,'gini003',.003,0.,32)
        self.steps=steps;self.power=power
        self.capture_stream=torch.cuda.Stream();self.ensure=MethodType(ensure_owned_stream,self)
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        probability=F.softmax(self.logits[:length-1],dim=-1)
        soft=BF16Mixture.apply(probability,self.weight)
        ids=self.logits[:length-1].argmax(-1)
        if self.alpha:
            alpha=torch.ones((),device="cuda")*self.alpha if not self.start else self.alpha*((self.counter[0]-self.start)/32.).clamp(0,1)
            soft=straight_through_mix(soft,self.weight[ids].float(),alpha)
        predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        purity=1-probability.square().sum(-1)
        strength=self.strength*(self.counter[0]>=self.start)
        base=error.mean()
        if self.power!=1:
            focused=error.clamp_min(0).pow(self.power).mean()/(self.power*.05**(self.power-1))
            base=torch.where(self.counter[0]>=32,focused,base)
        loss=base+strength*purity.mean()
        with torch.no_grad():
            eligible=self.counter[0]>=self.final_phase
            self.tokens[1:length].copy_(ids)
            improved=(loss.detach()<self.best_loss)&eligible
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.where(eligible,torch.minimum(self.best_loss,loss.detach()),self.best_loss))
            observed=error.detach().mean()
            improved_error=(observed<self.best_error)&eligible
            self.best_error_tokens[1:length].copy_(torch.where(improved_error,ids,self.best_error_tokens[1:length]))
            self.best_error.copy_(torch.where(eligible,torch.minimum(self.best_error,observed),self.best_error))
            per=error.detach();better=(per<self.position_best_loss[:length-1])&eligible
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.where(eligible,torch.minimum(self.position_best_loss[:length-1],per),self.position_best_loss[:length-1]))
            self.position_error[:length-1].copy_(per)
            confidence=probability.detach().amax(-1)
            self.confidence[:length-1].copy_(confidence)
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.error_trace.scatter_(0,self.counter,observed.reshape(1))
            self.confidence_trace.scatter_(0,self.counter,confidence.mean().reshape(1))
            self.purity_trace.scatter_(0,self.counter,purity.detach().mean().reshape(1))
            self.counter.add_(1)
        return loss

class FocusReconstruction:
    def __init__(self,prefix,steps,power):self.soft=FocusSoft(prefix,steps,power)
    @property
    def capture_events(self):return self.soft.capture_events
    def decode(self,h):
        out,stats=self.soft.decode(h)
        return out,{'total_seconds':stats['total_seconds'],'soft':stats,'power':self.soft.power,
          'focus_starts_at':32,'reference_error':.05,'whole_sequence_prefix_forwards':self.soft.steps+1,
          'whole_sequence_prefix_backwards':self.soft.steps,'vocabulary_entries_per_sweep':128256,
          'shortlist_size':None,'separate_candidate_verification_calls':0,'model_parameter_updates':0,
          'decision':'unchanged final-logit argmax; full vocabulary input optimization'}

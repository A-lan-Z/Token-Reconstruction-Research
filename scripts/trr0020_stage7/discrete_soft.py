"""Full-vocabulary optimization with delayed discreteness constraints."""
import torch
from torch.nn import functional as F
from fast_soft import FastSoftVocabulary,BF16Mixture

CONFIGS=[
 ("control",0.,0.,0),
 ("gini003",.003,0.,32),
 ("gini01",.01,0.,32),
 ("gini03",.03,0.,32),
 ("hardhalf64",0.,.5,64),
 ("hardfull64",0.,1.,64),
 ("hardfull128",0.,1.,128),
 ("hardfull0",0.,1.,0),
]

def straight_through_mix(soft,hard,alpha):
    value=(1-alpha)*soft.detach()+alpha*hard
    return value+(soft-soft.detach())

class DiscreteSoftVocabulary(FastSoftVocabulary):
    def __init__(self,prefix,name,strength,alpha,start):
        super().__init__(prefix,"adaptive",256)
        self.name=name;self.strength=strength;self.alpha=alpha;self.start=start
        self.final_phase=start+32 if alpha and start else start
        self.best_error=torch.tensor(float("inf"),device="cuda")
        self.best_error_tokens=self.tokens.clone()
        self.confidence=torch.zeros(127,device="cuda")
        self.error_trace=torch.zeros(1025,device="cuda")
        self.confidence_trace=torch.zeros(1025,device="cuda")
        self.purity_trace=torch.zeros(1025,device="cuda")
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
        loss=error.mean()+strength*purity.mean()
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
    @torch.no_grad()
    def reset(self,h):
        super().reset(h)
        self.best_error.fill_(float("inf"));self.best_error_tokens.fill_(128000)
        self.confidence.zero_();self.error_trace.zero_();self.confidence_trace.zero_();self.purity_trace.zero_()
    def decode(self,observations):
        outputs,stats=super().decode(observations)
        length=len(observations)
        outputs["best_objective"]=outputs.pop("best_soft_objective")
        outputs["best_observed_error"]=self.best_error_tokens[:length].cpu()
        stats.update(discreteness_rule={"name":self.name,"gini_strength":self.strength,"hard_alpha":self.alpha,"start":self.start,"best_selection_after":self.final_phase},
          observed_error_trace=self.error_trace[:self.steps+1].cpu().tolist(),
          mean_confidence_trace=self.confidence_trace[:self.steps+1].cpu().tolist(),
          mean_gini_trace=self.purity_trace[:self.steps+1].cpu().tolist(),
          final_position_confidence=self.confidence[:length-1].cpu().tolist(),
          final_position_error=self.position_error[:length-1].cpu().tolist())
        return outputs,stats

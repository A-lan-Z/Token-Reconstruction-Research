"""Full-vocabulary optimization with a preconditioned softmax gradient."""
import torch
from torch.nn import functional as F
from fast_soft import FastSoftVocabulary,BF16Mixture

CONFIGS=[("control",0.,.6)]+[(f"power{str(power).replace('.','p')}_lr{str(lr).replace('.','p')}",power,lr)
 for power in [.5,.75,1.] for lr in [.15,.3,.6,1.2]]

class PreconditionedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,power):
        probability=F.softmax(logits,dim=-1)
        ctx.save_for_backward(probability);ctx.power=power
        return probability
    @staticmethod
    def backward(ctx,gradient):
        (probability,)=ctx.saved_tensors
        centered=gradient-(gradient*probability).sum(-1,keepdim=True)
        return centered*probability.pow(1-ctx.power),None

class NaturalSoftVocabulary(FastSoftVocabulary):
    def __init__(self,prefix,name,power,lr):
        super().__init__(prefix,"adaptive",128)
        self.name=name;self.power=power;self.base_lr=lr
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        probability=F.softmax(self.logits[:length-1],dim=-1) if self.power==0 else PreconditionedSoftmax.apply(self.logits[:length-1],self.power)
        soft=BF16Mixture.apply(probability,self.weight)
        predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        loss=error.mean()
        with torch.no_grad():
            ids=self.logits[:length-1].argmax(-1)
            self.tokens[1:length].copy_(ids)
            improved=loss.detach()<self.best_loss
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,loss.detach()))
            per=error.detach();better=per<self.position_best_loss[:length-1]
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.minimum(self.position_best_loss[:length-1],per))
            self.position_error[:length-1].copy_(per)
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1));self.counter.add_(1)
        return loss
    @torch.no_grad()
    def reset(self,h):
        super().reset(h);self.lr_tensor.fill_(self.base_lr)
    def decode(self,observations):
        out,stats=super().decode(observations)
        stats.update(gradient_preconditioner_power=self.power,base_lr=self.base_lr,method=self.name)
        return out,stats

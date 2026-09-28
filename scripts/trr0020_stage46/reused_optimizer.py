"""Reuse the forward probability in the unchanged exact full-vocabulary update."""
import torch
from torch.nn import functional as F
from exact_optimizer import ExactOptimizer
from natural_soft import PreconditionedSoftmax
from fast_soft import BF16Mixture
from reused_step import update
class ReusedOptimizer(ExactOptimizer):
    def __init__(self,prefix,reuse):
        self.reuse=reuse;self.reused_probability=None
        super().__init__(prefix,True)
    def evaluate(self,length):
        if self.stage=="warm" or not self.reuse:return super().evaluate(length)
        pe,mask=self.geometry[length]
        p=PreconditionedSoftmax.apply(self.logits[:length-1],1.)
        self.reused_probability=p.detach()
        soft=BF16Mixture.apply(p,self.weight)
        predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        loss=error.mean()
        with torch.no_grad():
            ids=self.logits[:length-1].argmax(-1);self.tokens[1:length].copy_(ids)
            improved=loss.detach()<self.best_loss
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,loss.detach()))
            per=error.detach();better=per<self.position_best_loss[:length-1]
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.minimum(self.position_best_loss[:length-1],per))
            self.position_error[:length-1].copy_(per)
            confidence=p.detach().amax(-1);self.confidence[:length-1].copy_(confidence)
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.error_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.confidence_trace.scatter_(0,self.counter,confidence.mean().reshape(1))
            self.purity_trace.scatter_(0,self.counter,(1-p.detach().square().sum(-1)).mean().reshape(1))
            self.counter.add_(1)
        return loss
    def train_step(self,length):
        if self.stage=="warm" or not self.reuse:return super().train_step(length)
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        with torch.no_grad():
            new,info=update(self.logits[:length-1],self.gradient[:length-1],self.position_error[:length-1],2.,4,probability=self.reused_probability)
            self.logits[:length-1].copy_(new)
            record=torch.stack([info["requested_budget"].mean(),info["formula_kl"].mean(),info["budget_used"].mean(),
              info["normalized_step"].amax(),info["active"].float().mean()])
            self.mirror_update_trace.index_copy_(0,self.counter-1,record[None])
    def decode(self,h,steps=64,replay=True):
        out,stats=super().decode(h,steps,replay)
        stats["update_implementation"]="reused_probability_pointwise_fp32" if self.reuse else "exact_pointwise_fp32"
        return out,stats

def cache_reference(engine,h,data):
    """Independent embedding cotangent and frozen original-state anchors."""
    from pointwise import bits_equal
    length=len(h)
    engine.stage="warm";engine.reset(h);engine.reset_mirror_statistics()
    with torch.no_grad():engine.logits[:length-1].copy_(data["logits"].to("cuda"))
    engine.stage="mirror";pe,mask=engine.geometry[length]
    with torch.no_grad():
        probability=F.softmax(engine.logits[:length-1],dim=-1)
        mean=torch.mm(probability.to(torch.bfloat16),engine.weight,out_dtype=torch.float32)
    variable=mean.detach().requires_grad_()
    predicted=engine.forward(torch.cat([engine.bos,variable[None]],dim=1),pe,mask)
    error=(1-F.cosine_similarity(predicted[:,1:],engine.h[None,1:length],dim=-1))[0]
    loss=error.mean()
    cotangent=torch.autograd.grad(loss,variable)[0]
    with torch.no_grad():
        raw=torch.mm(cotangent.to(torch.bfloat16),engine.weight.T,out_dtype=torch.float32)
        expected=raw-(raw*probability).sum(-1,keepdim=True)
    engine.gradient.zero_();actual_loss=engine.evaluate(length);actual_loss.backward()
    got=engine.gradient[:length-1]
    checks={"probability_bytes_equal":bits_equal(engine.reused_probability,probability),
      "gradient_bytes_equal":bits_equal(got,expected),
      "frozen_gradient_bytes_equal":bits_equal(got.cpu(),data["gradient"]),
      "frozen_error_bytes_equal":bits_equal(engine.position_error[:length-1].cpu(),data["error"]),
      "loss_bytes_equal":bits_equal(loss,actual_loss)}
    arrays={"probability":engine.reused_probability.detach().cpu(),"gradient":got.detach().cpu(),
      "error":engine.position_error[:length-1].detach().cpu()}
    checks["passed"]=all(checks.values());engine.stage="warm"
    return checks,arrays

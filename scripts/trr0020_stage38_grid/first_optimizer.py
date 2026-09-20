"""Joint full-vocabulary inference with an optional final first-token decision."""
import time,torch
from torch.nn import functional as F
from budget_optimizer import BudgetOptimizer,CHECKPOINTS
from discrete_soft import DiscreteSoftVocabulary
from natural_soft import PreconditionedSoftmax
from fast_soft import BF16Mixture
from lookup import select

class FirstOptimizer(BudgetOptimizer):
    def __init__(self,prefix,locked,table=None):
        self.locked=locked
        super().__init__(prefix,2.,4)
        self.response_table=table
        self.response_lengths=table.norm(dim=-1) if locked else None
        self.first_id=torch.full((1,),128000,device="cuda",dtype=torch.long)
        self.lookup_seconds=0.
    @torch.no_grad()
    def reset(self,h):
        super().reset(h)
        if self.locked:
            torch.cuda.synchronize();t=time.perf_counter()
            chosen,_=select(self.response_table,self.response_lengths,self.h[1])
            self.first_id.copy_(chosen);torch.cuda.synchronize();self.lookup_seconds=time.perf_counter()-t
        else:self.lookup_seconds=0.
    def evaluate(self,length):
        if self.stage=="warm" or not self.locked:return super().evaluate(length)
        pe,mask=self.geometry[length]
        p=PreconditionedSoftmax.apply(self.logits[:length-1],1.)
        soft=BF16Mixture.apply(p,self.weight)
        fixed=self.weight.index_select(0,self.first_id).float()
        actual=torch.cat([fixed,soft[1:]],dim=0)
        predicted=self.forward(torch.cat([self.bos,actual[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        loss=error.mean()
        with torch.no_grad():
            ids=self.logits[:length-1].argmax(-1);ids[:1].copy_(self.first_id)
            self.tokens[1:length].copy_(ids)
            improved=loss.detach()<self.best_loss
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,loss.detach()))
            per=error.detach();better=per<self.position_best_loss[:length-1]
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.minimum(self.position_best_loss[:length-1],per))
            self.position_error[:length-1].copy_(per)
            confidence=p.detach().amax(-1);confidence[:1].fill_(1.)
            purity=1-p.detach().square().sum(-1);purity[:1].zero_()
            self.confidence[:length-1].copy_(confidence)
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.error_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.confidence_trace.scatter_(0,self.counter,confidence.mean().reshape(1))
            self.purity_trace.scatter_(0,self.counter,purity.mean().reshape(1))
            self.counter.add_(1)
        return loss
    def decode(self,h,steps=64,replay=True):
        self.stage="warm";torch.cuda.synchronize();start=time.perf_counter();length=len(h)
        warm_capture=self.ensure(length);mirror_capture=self.ensure_mirror(length)
        warm,ws=DiscreteSoftVocabulary.decode(self,h);torch.cuda.synchronize();warm_end=time.perf_counter()
        with torch.no_grad():
            p=F.softmax(self.logits[:length-1],dim=-1);mean=BF16Mixture.apply(p,self.weight)
            out={"warm_"+k:v for k,v in warm.items()}
            if self.locked:
                out["base_initial_embedding"]=mean.cpu()
                mean=torch.cat([self.weight.index_select(0,self.first_id).float(),mean[1:]],0)
                out["first_choice"]=self.first_id.cpu()
            out["initial_embedding"]=mean.cpu()
            self.reset_mirror_statistics()
        self.stage="mirror";torch.cuda.synchronize();mirror_start=time.perf_counter();checkpoints={}
        training,evaluating=self.mirror_graphs[length]
        for step in range(steps):
            if replay:training.replay()
            else:self.train_step(length)
            if step in CHECKPOINTS:
                out["step"+str(step)]=self.tokens[:length].cpu()
                torch.cuda.synchronize();checkpoints[str(step)]=time.perf_counter()-start
        if replay:evaluating.replay()
        else:
            with torch.no_grad():self.evaluate(length)
        with torch.no_grad():
            out["step"+str(steps)]=self.tokens[:length].cpu()
            out["best_objective"]=self.best_tokens[:length].cpu()
            out["best_position_error"]=self.position_best_tokens[:length].cpu()
            out["mirror_update_trace"]=self.mirror_update_trace[:steps].cpu()
            out["final_confidence"]=self.confidence[:length-1].cpu()
            out["final_position_error"]=self.position_error[:length-1].cpu()
            traces={"loss":self.loss_trace[:steps+1].cpu().tolist(),"confidence":self.confidence_trace[:steps+1].cpu().tolist(),
                    "gini":self.purity_trace[:steps+1].cpu().tolist()}
            valid=bool(torch.isfinite(self.logits[:length-1]).all()) and all(bool(torch.isfinite(v).all()) for v in out.values())
        torch.cuda.synchronize();end=time.perf_counter();checkpoints[str(steps)]=end-start;self.stage="warm"
        return out,{"total_seconds":end-start,"warm":ws,"warm_capture_seconds":warm_capture,"mirror_capture_seconds":mirror_capture,
          "warm_stage_seconds_including_capture":warm_end-start,"mirror_seconds":end-mirror_start,
          "mixture_transfer_and_reset_seconds":mirror_start-warm_end,"checkpoint_total_seconds":checkpoints,
          "mirror_traces":traces,"mirror_steps":steps,"factor":2.,"scalar_iterations":4,"first_locked":self.locked,
          "first_lookup_seconds":self.lookup_seconds,"numeric_valid":valid,"whole_sequence_prefix_forwards":steps+2,
          "whole_sequence_prefix_backwards":steps,"additional_mixture_products":1,"vocabulary_entries_per_sweep":self.vocab,
          "first_lookup_vocabulary":self.vocab if self.locked else 0,"shortlist_size":None,"separate_candidate_verification_calls":0,
          "model_parameter_updates":0,"budget_rule":"clamp(factor * current observed cosine error,0,1)",
          "timing_scope":"native full geometry; actual stopping budget; input initialization and optionalfirstlookup, updates, diagnostics, synchronization and output transfer included; capture/preparation separate"}

def gradient_reference(engine,h):
    engine.reset(h);engine.reset_mirror_statistics();engine.stage="mirror";length=len(h)
    with torch.no_grad():
        probability=F.softmax(engine.logits[:length-1],dim=-1)
        mean=torch.mm(probability.to(torch.bfloat16),engine.weight,out_dtype=torch.float32)
    variable=mean.detach().requires_grad_();pe,mask=engine.geometry[length]
    actual=torch.cat([engine.weight.index_select(0,engine.first_id).float(),variable[1:]],0) if engine.locked else variable
    predicted=engine.forward(torch.cat([engine.bos,actual[None]],dim=1),pe,mask)
    loss=(1-F.cosine_similarity(predicted[:,1:],engine.h[None,1:length],dim=-1)).mean()
    cotangent=torch.autograd.grad(loss,variable)[0]
    with torch.no_grad():
        raw=torch.mm(cotangent.to(torch.bfloat16),engine.weight.T,out_dtype=torch.float32)
        expected=raw-(raw*probability).sum(-1,keepdim=True)
    engine.gradient.zero_();actual_loss=engine.evaluate(length);actual_loss.backward()
    got=engine.gradient[:length-1]
    result={"loss_equal":torch.equal(loss,actual_loss),"gradient_exact":torch.equal(got,expected),
      "maximum_gradient_error":float((got-expected).abs().max()),
      "first_gradient_zero":bool((got[0]==0).all()) if engine.locked else None,
      "scope":"independent embedding cotangent then declared BF16 probability-gradient product, not finite differences through quantization"}
    result["passed"]=result["loss_equal"] and result["gradient_exact"] and (not engine.locked or result["first_gradient_zero"])
    engine.stage="warm";return result

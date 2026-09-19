"""Whole-vocabulary mirror descent after an unchanged optional Gini warm start."""
from collections import OrderedDict
from types import MethodType
import time,torch
from torch.nn import functional as F
from discrete_soft import DiscreteSoftVocabulary
from natural_soft import PreconditionedSoftmax
from fast_soft import BF16Mixture
from reuse_stream import ensure_owned_stream
from mirror_step import update
CONFIGS=[(f"warm{warm}_tau{tau}",warm,tau) for warm in [0,64] for tau in [.01,.1,1.]]
CHECKPOINTS=(0,8,16,32,64)
AUX_KEYS={"initial_embedding","mirror_update_trace","final_confidence","final_position_error"}

class MirrorOptimizer(DiscreteSoftVocabulary):
    def __init__(self,prefix,warm,tau):
        super().__init__(prefix,"gini003",.003,0.,32)
        self.steps=warm;self.tau=tau;self.stage="warm";self.mirror_steps=64
        self.capture_stream=torch.cuda.Stream()
        self.mirror_graphs=OrderedDict();self.mirror_capture_events=[]
        self.mirror_update_trace=torch.zeros((1025,5),device="cuda")
        self.ensure=MethodType(self.owned_ensure.__func__,self)
    def owned_ensure(self,length):
        if length not in self.graphs and len(self.graphs)>=2:
            old=next(iter(self.graphs));self.mirror_graphs.pop(old,None)
        return ensure_owned_stream(self,length)
    def evaluate(self,length):
        if self.stage=="warm":return super().evaluate(length)
        pe,mask=self.geometry[length]
        p=PreconditionedSoftmax.apply(self.logits[:length-1],1.)
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
        if self.stage=="warm":return super().train_step(length)
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        with torch.no_grad():
            new,info=update(self.logits[:length-1],self.gradient[:length-1],self.tau)
            self.logits[:length-1].copy_(new)
            record=torch.stack([info["rate"].mean(),info["weighted_variance"].mean(),
              info["gradient_span"].mean(),info["update_span"].mean(),info["update_span"].amax()])
            self.mirror_update_trace.index_copy_(0,self.counter-1,record[None])
    @torch.no_grad()
    def reset_mirror_statistics(self):
        self.counter.zero_();self.loss_trace.zero_();self.error_trace.zero_()
        self.confidence_trace.zero_();self.purity_trace.zero_();self.mirror_update_trace.zero_()
        self.best_loss.fill_(float("inf"));self.position_best_loss.fill_(float("inf"))
        self.tokens.fill_(128000);self.best_tokens.fill_(128000);self.position_best_tokens.fill_(128000)
        self.position_error.zero_();self.confidence.zero_()
    def ensure_mirror(self,length):
        if length in self.mirror_graphs:
            self.mirror_graphs.move_to_end(length);return 0.
        torch.cuda.synchronize();start=time.perf_counter()
        self.stage="mirror"
        with torch.no_grad():
            dummy=self.prefix.forward_full(torch.full((1,length),128000,device="cuda",dtype=torch.long))[0].float()
            self.reset(dummy);self.reset_mirror_statistics()
        stream=self.capture_stream;stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.train_step(length)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        training=torch.cuda.CUDAGraph()
        with torch.cuda.graph(training,stream=stream):self.train_step(length)
        evaluating=torch.cuda.CUDAGraph()
        with torch.cuda.graph(evaluating,stream=stream):
            with torch.no_grad():self.evaluate(length)
        torch.cuda.synchronize();self.mirror_graphs[length]=(training,evaluating);self.stage="warm"
        elapsed=time.perf_counter()-start
        self.mirror_capture_events.append({"length":length,"seconds":elapsed,"reserved_bytes":torch.cuda.memory_reserved()})
        return elapsed
    def decode(self,h,replay=True):
        self.stage="warm";torch.cuda.synchronize();start=time.perf_counter();length=len(h)
        warm_capture=self.ensure(length);mirror_capture=self.ensure_mirror(length)
        warm,ws=super().decode(h);torch.cuda.synchronize();warm_end=time.perf_counter()
        with torch.no_grad():
            p=F.softmax(self.logits[:length-1],dim=-1)
            mean=BF16Mixture.apply(p,self.weight)
            out={"warm_"+k:v for k,v in warm.items()};out["initial_embedding"]=mean.cpu()
            self.reset_mirror_statistics()
        self.stage="mirror";torch.cuda.synchronize();mirror_start=time.perf_counter();checkpoints={}
        training,evaluating=self.mirror_graphs[length]
        for step in range(self.mirror_steps):
            if replay:training.replay()
            else:self.train_step(length)
            if step in CHECKPOINTS:
                out["step"+str(step)]=self.tokens[:length].cpu()
                torch.cuda.synchronize();checkpoints[str(step)]=time.perf_counter()-start
        if replay:evaluating.replay()
        else:
            with torch.no_grad():self.evaluate(length)
        with torch.no_grad():
            out["step64"]=self.tokens[:length].cpu()
            out["best_objective"]=self.best_tokens[:length].cpu()
            out["best_position_error"]=self.position_best_tokens[:length].cpu()
            out["mirror_update_trace"]=self.mirror_update_trace[:64].cpu()
            out["final_confidence"]=self.confidence[:length-1].cpu()
            out["final_position_error"]=self.position_error[:length-1].cpu()
            traces={"loss":self.loss_trace[:65].cpu().tolist(),"confidence":self.confidence_trace[:65].cpu().tolist(),
              "gini":self.purity_trace[:65].cpu().tolist()}
            valid=bool(torch.isfinite(self.logits[:length-1]).all()) and all(bool(torch.isfinite(v).all()) for v in out.values())
        torch.cuda.synchronize();end=time.perf_counter();checkpoints["64"]=end-start;self.stage="warm"
        return out,{"total_seconds":end-start,"warm":ws,"warm_capture_seconds":warm_capture,"mirror_capture_seconds":mirror_capture,
          "warm_stage_seconds_including_capture":warm_end-start,"mirror_seconds":end-mirror_start,
          "mixture_transfer_and_reset_seconds":mirror_start-warm_end,"checkpoint_total_seconds":checkpoints,
          "mirror_traces":traces,"mirror_steps":64,"tau":self.tau,"max_update_span":4.,
          "numeric_valid":valid,"whole_sequence_prefix_forwards":self.steps+1+65,"whole_sequence_prefix_backwards":self.steps+64,
          "additional_mixture_products":1,"vocabulary_entries_per_sweep":self.vocab,"shortlist_size":None,
          "separate_candidate_verification_calls":0,"model_parameter_updates":0,
          "update_trace_columns":["mean_rate","mean_weighted_variance","mean_gradient_span","mean_update_span","max_update_span"],
          "numerics":"existing BF16 surrogate mixture gradient; power1 backward; direct FP32 mirror step, no Adam or decay; TF32 disabled",
          "timing_scope":"warm plus64mirror steps,65mirror evaluations, diagnostic outputs, synchronization and transfers; first-input captures included and separately recorded"}

def gradient_reference(engine,h):
    """Independent embedding cotangent times E^T, without softmax custom backward."""
    length=len(h);engine.stage="mirror"
    with torch.no_grad():
        probability=F.softmax(engine.logits[:length-1],dim=-1)
        mean=torch.mm(probability.to(torch.bfloat16),engine.weight,out_dtype=torch.float32)
    variable=mean.detach().requires_grad_();pe,mask=engine.geometry[length]
    predicted=engine.forward(torch.cat([engine.bos,variable[None]],dim=1),pe,mask)
    loss=(1-F.cosine_similarity(predicted[:,1:],engine.h[None,1:length],dim=-1)).mean()
    cotangent=torch.autograd.grad(loss,variable)[0]
    with torch.no_grad():
        raw=torch.mm(cotangent.to(torch.bfloat16),engine.weight.T,out_dtype=torch.float32)
        expected=raw-(raw*probability).sum(-1,keepdim=True)
    engine.gradient.zero_();actual_loss=engine.evaluate(length);actual_loss.backward()
    actual=engine.gradient[:length-1]
    positions=torch.tensor([0,(length-1)//2,length-2],device="cuda")
    tokens=torch.tensor([0,7,engine.vocab//2,engine.vocab-1],device="cuda")
    result={"loss_equal":torch.equal(loss,actual_loss),"gradient_exact":torch.equal(actual,expected),
      "max_absolute_error":float((actual-expected).abs().max()),
      "actual_sample":actual[positions[:,None],tokens[None]].detach().cpu().tolist(),
      "expected_sample":expected[positions[:,None],tokens[None]].cpu().tolist(),
      "scope":"declared BF16 surrogate probability gradient via independent embedding cotangent; not finite differences through quantization"}
    engine.stage="warm";result["passed"]=result["loss_equal"] and result["gradient_exact"]
    return result

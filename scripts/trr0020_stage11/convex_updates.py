"""Full-vocabulary conditional-gradient inversion without dense logit gradients."""
import time
import torch
from torch.nn import functional as F
from proximal import ProximalVocabulary

CONFIGS=[(f"rate{str(rate).replace('.','p')}_adapt{int(adapt)}_anneal{int(anneal)}",rate,adapt,anneal)
 for rate in [.03,.1,.3] for adapt in [False,True] for anneal in [False,True]]

def reference_test():
    torch.manual_seed(200031)
    e=torch.randn(29,13,dtype=torch.float64)
    p=torch.softmax(torch.randn(5,29,dtype=torch.float64),dim=-1);z=p@e
    maximum=0.
    for step in range(128):
        gradient=torch.randn_like(z)
        scores=gradient@e.T;ids=scores.argmin(-1)
        alpha=torch.rand(5,1,dtype=torch.float64)*.3
        direction=e[ids]-z
        explicit=(gradient[:,None]*(e[None]-z[:,None])).sum(-1)
        assert torch.equal(ids,explicit.argmin(-1))
        p=p*(1-alpha);p.scatter_add_(1,ids[:,None],alpha)
        z=z+alpha*direction
        torch.testing.assert_close(z,p@e,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(p.sum(-1),torch.ones(5,dtype=torch.float64),rtol=1e-12,atol=1e-12)
        assert p.min()>=0
        maximum=max(maximum,float((z-p@e).abs().max()))
    return {"steps":128,"convex_embedding_identity_max_error":maximum,"linear_minimization_verified":True,"distribution_normalization_verified":True}

class ConvexVocabulary(ProximalVocabulary):
    def __init__(self,prefix,rate,adaptive,anneal):
        super().__init__(prefix,"raw",1.)
        self.steps=128;self.rate=rate;self.adaptive=adaptive;self.anneal=anneal
        self.probability=torch.zeros((127,len(self.weight)),device="cuda")
        self.trace=torch.zeros(129,device="cuda")
    @torch.no_grad()
    def reset(self,h):
        length=len(h);self.h[:length].copy_(h.to("cuda").float())
        q=F.normalize(self.h[1:length]@self.metric.transform,dim=-1)
        scores=q@self.metric.table.T
        self.probability.zero_();self.probability[:length-1].copy_(F.softmax(80*scores,dim=-1))
        self.z.zero_()
        self.z[:length-1].copy_(torch.mm(self.probability[:length-1].to(torch.bfloat16),self.weight,out_dtype=torch.float32))
        self.tokens.fill_(128000);self.tokens[1:length].copy_(self.probability[:length-1].argmax(-1))
        self.gradient.zero_();self.best_tokens.fill_(128000);self.position_tokens.fill_(128000)
        self.best_loss.fill_(float("inf"));self.position_best.fill_(float("inf"));self.counter.zero_();self.trace.zero_();self.error.zero_()
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        predicted=self.forward(torch.cat([self.bos,self.z[None,:length-1]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        with torch.no_grad():
            self.tokens[1:length].copy_(self.probability[:length-1].argmax(-1))
            mean=error.mean();improved=mean<self.best_loss
            self.best_tokens[:length].copy_(torch.where(improved,self.tokens[:length],self.best_tokens[:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,mean))
            better=error<self.position_best[:length-1]
            self.position_tokens[1:length].copy_(torch.where(better,self.tokens[1:length],self.position_tokens[1:length]))
            self.position_best[:length-1].copy_(torch.minimum(self.position_best[:length-1],error))
            self.error[:length-1].copy_(error);self.trace.scatter_(0,self.counter,mean.reshape(1));self.counter.add_(1)
        return error.sum()
    def train_step(self,length):
        self.gradient.zero_();self.evaluate(length).backward()
        with torch.no_grad():
            g=self.gradient[:length-1]
            scores=torch.mm(g.to(torch.bfloat16),self.weight.T,out_dtype=torch.float32)
            ids=scores.argmin(-1)
            rate=self.rate/(1+(self.counter[0]-1)/32.) if self.anneal else self.rate
            alpha=torch.ones((length-1,1),device="cuda")*rate
            if self.adaptive:alpha=alpha*(self.error[:length-1,None]/.05).clamp(0,1)
            self.z[:length-1].add_(alpha*(self.weight[ids].float()-self.z[:length-1]))
            self.probability[:length-1].mul_(1-alpha)
            self.probability[:length-1].scatter_add_(1,ids[:,None],alpha)
    def decode(self,observations):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter();length=len(observations)
        capture=self.ensure(length);self.reset(observations);torch.cuda.synchronize();prepared=time.perf_counter()
        train,evaluate=self.graphs[length];outputs={}
        for i in range(self.steps):
            train.replay()
            if i in {0,16,32,64}:outputs["step"+str(i)]=self.tokens[:length].cpu()
        evaluate.replay()
        outputs["step128"]=self.tokens[:length].cpu();outputs["best_objective"]=self.best_tokens[:length].cpu()
        outputs["best_position_error"]=self.position_tokens[:length].cpu()
        loss=self.trace.cpu();normalization=(self.probability[:length-1].sum(-1)-1).abs().max().cpu()
        if not torch.isfinite(loss).all() or normalization>1e-4:raise RuntimeError("nonfinite or invalid convex distribution")
        torch.cuda.synchronize();end=time.perf_counter()
        return outputs,{"total_seconds":end-start,"capture_seconds":capture,"input_initialization_seconds":prepared-start-capture,
          "optimization_output_seconds":end-prepared,"loss_trace":loss.tolist(),"probability_normalization_error":float(normalization),
          "whole_sequence_prefix_forwards":129,"whole_sequence_prefix_backwards":128,"full_vocabulary_sweeps":129,
          "vocabulary_entries_per_sweep":len(self.weight),"shortlist_size":None,"separate_candidate_verification_calls":0,
          "input_optimization_steps":128,"model_parameter_updates":0,"decision":"argmax of full-vocabulary convex weights",
          "update":"minimize linearized error over every vocabulary embedding, then interpolate",
          "rate":self.rate,"adaptive":self.adaptive,"anneal":self.anneal,"score_numerics":"BF16query_embeddings_FP32output",
          "prefix_geometry":"native_length_no_padding"}

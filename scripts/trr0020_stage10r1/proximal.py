"""Direct whole-vocabulary quadratic updates from one A2 gradient."""
from collections import OrderedDict
import gc,time
import torch
from torch.nn import functional as F
from soft_vocabulary import SoftVocabulary
from discrete_parallel import DiscreteParallel

CONFIGS=[(f"{metric}_beta{str(beta).replace('.','p')}",metric,beta) for metric in ["raw","white"] for beta in [.003,.01,.03,.1,.25]]

def reference_test():
    torch.manual_seed(200030)
    e=torch.randn(23,7,dtype=torch.float64);z=torch.randn(4,7,dtype=torch.float64)
    g=torch.randn_like(z);r=torch.randn(7,7,dtype=torch.float64)
    a=torch.tensor([.2,.5,1.,2.],dtype=torch.float64)[:,None]
    norm=(e@r).square().sum(-1)
    score=(a*(z@r@r.T)-g)@e.T-.5*a*norm
    delta=e[None]-z[:,None]
    reference=-(g[:,None]*delta).sum(-1)-.5*a*(delta@r).square().sum(-1)
    torch.testing.assert_close(score-score[:,:1],reference-reference[:,:1],rtol=1e-12,atol=1e-12)
    target=e[torch.tensor([0,7,15,22])];gradient=z-target
    exact=(z-gradient)@e.T-.5*e.square().sum(-1)
    assert torch.equal(exact.argmax(-1),torch.tensor([0,7,15,22]))
    return {"quadratic_identity_max_error":float(((score-score[:,:1])-(reference-reference[:,:1])).abs().max()),
      "linear_identity_inverse_recovers_all_including_last_vocabulary_entry":True}

class ProximalVocabulary(SoftVocabulary):
    def __init__(self,prefix,metric,beta):
        super().__init__(prefix)
        self.bos=prefix.embed_tokens.weight[128000].detach().float().clone().view(1,1,-1)
        self.weight=prefix.embed_tokens.weight.detach()
        self.kind=metric;self.beta=beta;self.steps=64
        self.r=self.transform if metric=="white" else torch.eye(2048,device="cuda")
        self.rinv=torch.linalg.inv(self.r) if metric=="white" else self.r
        self.norm=torch.empty(len(self.weight),device="cuda")
        for lo in range(0,len(self.weight),4096):
            v=self.weight[lo:lo+4096].float()
            if metric=="white":v=v@self.r
            self.norm[lo:lo+4096]=v.square().sum(-1)
        self.z=torch.zeros((127,2048),device="cuda",requires_grad=True)
        self.gradient=torch.zeros_like(self.z);self.z.grad=self.gradient
        self.h=torch.zeros((128,2048),device="cuda")
        self.tokens=torch.full((128,),128000,device="cuda",dtype=torch.long)
        self.best_tokens=self.tokens.clone();self.position_tokens=self.tokens.clone()
        self.best_loss=torch.tensor(float("inf"),device="cuda")
        self.position_best=torch.full((127,),float("inf"),device="cuda")
        self.counter=torch.zeros(1,device="cuda",dtype=torch.long)
        self.trace=torch.zeros(65,device="cuda");self.error=torch.zeros(127,device="cuda")
        self.graphs=OrderedDict();self.geometry={};self.capture_events=[]
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        z=torch.cat([self.bos,self.z[None,:length-1]],dim=1)
        predicted=DiscreteParallel.diagonal_forward(self,z,pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        with torch.no_grad():
            mean=error.mean();improved=mean<self.best_loss
            self.best_tokens[:length].copy_(torch.where(improved,self.tokens[:length],self.best_tokens[:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,mean))
            better=error<self.position_best[:length-1]
            self.position_tokens[1:length].copy_(torch.where(better,self.tokens[1:length],self.position_tokens[1:length]))
            self.position_best[:length-1].copy_(torch.minimum(self.position_best[:length-1],error))
            self.error[:length-1].copy_(error)
            self.trace.scatter_(0,self.counter,mean.reshape(1));self.counter.add_(1)
        return error.sum()
    def train_step(self,length):
        self.gradient.zero_();self.evaluate(length).backward()
        with torch.no_grad():
            z=self.z[:length-1];g=self.gradient[:length-1]
            if self.kind=="white":
                dual=g@self.rinv.T;metric_z=(z@self.r)@self.r.T
            else:dual=g;metric_z=z
            curvature=(self.beta*dual.square().sum(-1)/(2*self.error[:length-1].clamp_min(1e-8))).clamp(1e-6,1e6)
            query=curvature[:,None]*metric_z-g
            scores=torch.mm(query.to(torch.bfloat16),self.weight.T,out_dtype=torch.float32)-.5*curvature[:,None]*self.norm
            ids=scores.argmax(-1)
            self.tokens[1:length].copy_(ids);self.z[:length-1].copy_(self.weight[ids].float())
    @torch.no_grad()
    def reset(self,h):
        length=len(h);self.h[:length].copy_(h.to("cuda").float())
        q=F.normalize(self.h[1:length]@self.metric.transform,dim=-1)
        ids=(q@self.metric.table.T).argmax(-1)
        self.tokens.fill_(128000);self.tokens[1:length].copy_(ids)
        self.z.zero_();self.z[:length-1].copy_(self.weight[ids].float())
        self.gradient.zero_();self.best_tokens.fill_(128000);self.position_tokens.fill_(128000)
        self.best_loss.fill_(float("inf"));self.position_best.fill_(float("inf"));self.counter.zero_();self.trace.zero_();self.error.zero_()
    def ensure(self,length):
        if length in self.graphs:self.graphs.move_to_end(length);return 0.
        torch.cuda.synchronize();start=time.perf_counter()
        if len(self.graphs)>=2:
            old,pair=self.graphs.popitem(last=False);del pair;self.geometry.pop(old);gc.collect();torch.cuda.empty_cache()
        with torch.no_grad():
            dummy=self.prefix.forward_full(torch.full((1,length),128000,device="cuda",dtype=torch.long))[0].float()
            pe=self.prefix.rotary_emb(dummy[None].to(torch.bfloat16),torch.arange(length,device="cuda").view(1,-1))
            mask=self.prefix._causal_mask(dummy[None].to(torch.bfloat16),start_pos=0,total_tokens=length)
            self.geometry[length]=(pe,mask);self.reset(dummy)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.train_step(length)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        training=torch.cuda.CUDAGraph()
        with torch.cuda.graph(training,stream=stream):self.train_step(length)
        evaluating=torch.cuda.CUDAGraph()
        with torch.cuda.graph(evaluating,stream=stream):
            with torch.no_grad():self.evaluate(length)
        torch.cuda.synchronize();self.graphs[length]=(training,evaluating)
        elapsed=time.perf_counter()-start
        self.capture_events.append({"length":length,"seconds":elapsed,"reserved_bytes":torch.cuda.memory_reserved(),"cached_lengths":list(self.graphs)})
        return elapsed
    def decode(self,observations):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter();length=len(observations)
        capture=self.ensure(length);self.reset(observations);torch.cuda.synchronize();prepared=time.perf_counter()
        train,evaluate=self.graphs[length];outputs={}
        for i in range(self.steps):
            if i in {0,1,2,4,8,16,32}:outputs["step"+str(i)]=self.tokens[:length].cpu()
            train.replay()
        evaluate.replay()
        outputs["step64"]=self.tokens[:length].cpu();outputs["best_objective"]=self.best_tokens[:length].cpu()
        outputs["best_position_error"]=self.position_tokens[:length].cpu()
        losses=self.trace.cpu()
        if not torch.isfinite(losses).all():raise RuntimeError("nonfinite proximal objective")
        torch.cuda.synchronize();end=time.perf_counter()
        return outputs,{"total_seconds":end-start,"capture_seconds":capture,"input_initialization_seconds":prepared-start-capture,
          "optimization_output_seconds":end-prepared,"loss_trace":losses.tolist(),"whole_sequence_prefix_forwards":65,
          "whole_sequence_prefix_backwards":64,"full_vocabulary_sweeps":65,"vocabulary_entries_per_sweep":len(self.weight),
          "shortlist_size":None,"separate_candidate_verification_calls":0,"input_optimization_steps":64,
          "model_parameter_updates":0,"decision":"direct argmax of full-vocabulary local quadratic","metric":self.kind,"beta":self.beta,
          "curvature":"beta*dual_gradient_norm_squared/(2*max(cosine_error,1e-8));clamped1e-6..1e6",
          "score_numerics":"BF16query_and_embeddings_FP32output","prefix_geometry":"native_length_no_padding"}

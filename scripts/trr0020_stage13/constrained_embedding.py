"""Embedding-sized input optimization with a full-vocabulary distance penalty."""
from collections import OrderedDict
import time
import torch
from torch.nn import functional as F
from soft_vocabulary import SoftVocabulary
from graph_soft import GraphSoftVocabulary
from fast_soft import BF16Mixture
from reset_soft import update_logits

CONFIGS=[(f"lambda{str(strength).replace('.','p')}_lr{str(lr).replace('.','p')}",strength,lr)
 for strength in [0.,.003,.01,.03] for lr in [.001,.003,.01]]

def reference_tests():
    torch.manual_seed(200034)
    table=torch.randn(23,7,dtype=torch.float64)
    z=torch.randn(5,7,dtype=torch.float64,requires_grad=True)
    direct=(z[:,None]-table[None]).square().sum(-1)
    factored=z.square().sum(-1,keepdim=True)-2*z@table.T+table.square().sum(-1)[None]
    torch.testing.assert_close(direct,factored,rtol=1e-12,atol=1e-12)
    selected=factored.detach().argmin(-1)
    scale=table.square().sum(-1).mean()
    objective=(z-table[selected]).square().sum(-1).mean()/scale
    gradient=torch.autograd.grad(objective,z)[0]
    brute_gradient=torch.autograd.grad(direct.min(-1).values.mean()/scale,z)[0]
    torch.testing.assert_close(gradient,brute_gradient,rtol=1e-12,atol=1e-12)
    exact=(table[:,None]-table[None]).square().sum(-1).argmin(-1)
    assert torch.equal(exact,torch.arange(len(table)))
    low=table.min(0).values;high=table.max(0).values
    assert torch.equal(table,table.clamp(min=low,max=high))
    return {"factored_distance_max_error":float((direct-factored).abs().max()),
      "nearest_distance_gradient_max_error":float((gradient-brute_gradient).abs().max()),
      "all_random_dictionary_entries_selectable":True,"coordinate_bounds_preserve_all_random_entries":True,
      "scope":"independent CPU float64 math only, not model reconstruction evidence"}

class ConstrainedEmbedding(SoftVocabulary):
    def __init__(self,prefix,strength,lr):
        super().__init__(prefix)
        self.bos=prefix.embed_tokens.weight[128000].detach().float().clone().view(1,1,-1)
        self.weight=prefix.embed_tokens.weight.detach()
        full=self.weight.float()
        self.norms=full.square().sum(-1)
        self.scale=self.norms.mean()
        self.lower=full.min(0).values;self.upper=full.max(0).values
        del full
        self.strength=strength;self.lr=lr;self.steps=128;self.tf32=False;self.maximum=128
        self.z=torch.zeros((127,2048),device="cuda",requires_grad=True)
        self.gradient=torch.zeros_like(self.z);self.z.grad=self.gradient
        self.moment=torch.zeros_like(self.z);self.variance=torch.zeros_like(self.z)
        self.h=torch.zeros((128,2048),device="cuda")
        self.tokens=torch.full((128,),128000,device="cuda",dtype=torch.long)
        self.best_tokens=self.tokens.clone();self.position_best_tokens=self.tokens.clone()
        self.best_loss=torch.tensor(float("inf"),device="cuda")
        self.position_best_loss=torch.full((127,),float("inf"),device="cuda")
        self.counter=torch.zeros(1,device="cuda",dtype=torch.long)
        self.loss_trace=torch.zeros(129,device="cuda");self.error_trace=torch.zeros(129,device="cuda")
        self.distance_trace=torch.zeros(129,device="cuda")
        self.graphs=OrderedDict();self.geometry={};self.capture_events=[]
    def evaluate(self,length):
        pe,mask=self.geometry[length];z=self.z[:length-1]
        with torch.no_grad():
            dot=BF16Mixture.apply(z,self.weight.T)
            ids=(2*dot-self.norms[None]).argmax(-1)
            center=self.weight[ids].float()
        predicted=self.forward(torch.cat([self.bos,z[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        distance=(z-center).square().sum(-1)/self.scale
        strength=self.strength*((self.counter[0]-32)/32.).clamp(0,1)
        loss=error.mean()+strength*distance.mean()
        with torch.no_grad():
            self.tokens[1:length].copy_(ids)
            eligible=self.counter[0]>=64
            improved=(loss.detach()<self.best_loss)&eligible
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.where(eligible,torch.minimum(self.best_loss,loss.detach()),self.best_loss))
            per=error.detach();better=(per<self.position_best_loss[:length-1])&eligible
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.where(eligible,torch.minimum(self.position_best_loss[:length-1],per),self.position_best_loss[:length-1]))
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.error_trace.scatter_(0,self.counter,error.detach().mean().reshape(1))
            self.distance_trace.scatter_(0,self.counter,distance.detach().mean().reshape(1))
            self.counter.add_(1)
        return loss
    def train_step(self,length):
        self.gradient.zero_();self.evaluate(length).backward()
        with torch.no_grad():
            update_logits(self.z,self.moment,self.variance,self.gradient,self.lr,1.)
            self.z.copy_(torch.maximum(torch.minimum(self.z,self.upper),self.lower))
    @torch.no_grad()
    def reset(self,h):
        length=len(h);self.h[:length].copy_(h.to("cuda").float())
        q=F.normalize(self.h[1:length]@self.metric.transform,dim=-1)
        probability=F.softmax(80*(q@self.metric.table.T),dim=-1)
        self.z.zero_();self.z[:length-1].copy_(BF16Mixture.apply(probability,self.weight))
        self.gradient.zero_();self.moment.zero_();self.variance.zero_()
        self.tokens.fill_(128000);self.best_tokens.fill_(128000);self.position_best_tokens.fill_(128000)
        self.best_loss.fill_(float("inf"));self.position_best_loss.fill_(float("inf"))
        self.counter.zero_();self.loss_trace.zero_();self.error_trace.zero_();self.distance_trace.zero_()
    def ensure(self,length):return GraphSoftVocabulary.ensure(self,length)
    def decode(self,observations):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter();length=len(observations)
        capture=self.ensure(length);self.reset(observations);torch.cuda.synchronize();prepared=time.perf_counter()
        training,evaluating=self.graphs[length];outputs={}
        for i in range(self.steps):
            training.replay()
            if i in {0,16,32,64}:outputs["step"+str(i)]=self.tokens[:length].cpu()
            if (i+1)%32==0:self.moment.zero_();self.variance.zero_()
        evaluating.replay()
        outputs["step128"]=self.tokens[:length].cpu()
        outputs["best_objective"]=self.best_tokens[:length].cpu()
        outputs["best_position_error"]=self.position_best_tokens[:length].cpu()
        loss=self.loss_trace.cpu()
        if not torch.isfinite(loss).all():raise RuntimeError("nonfinite constrained inverse")
        torch.cuda.synchronize();end=time.perf_counter()
        return outputs,{"total_seconds":end-start,"capture_seconds":capture,"input_initialization_seconds":prepared-start-capture,
          "optimization_output_seconds":end-prepared,"loss_trace":loss.tolist(),
          "observed_error_trace":self.error_trace.cpu().tolist(),"nearest_distance_trace":self.distance_trace.cpu().tolist(),
          "whole_sequence_prefix_forwards":129,"whole_sequence_prefix_backwards":128,
          "full_vocabulary_matrix_products":131,"vocabulary_entries_per_sweep":len(self.weight),
          "shortlist_size":None,"separate_candidate_verification_calls":0,"model_parameter_updates":0,
          "input_dimensions_per_position":2048,"regularization_strength":self.strength,"lr":self.lr,
          "regularization_schedule":"linear 0 to lambda between iterations32 and64",
          "nearest_numerics":"BF16 dot product, FP32 accumulation and norms; approximate Euclidean decision",
          "bounds":"coordinate minima/maxima of all native embeddings; every vocabulary row lies inside",
          "prefix_geometry":"native_length_no_padding"}

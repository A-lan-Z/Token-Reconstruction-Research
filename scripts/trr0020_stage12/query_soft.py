"""Full-vocabulary inversion parameterized by one embedding-sized query per position."""
from collections import OrderedDict
import time
import torch
from torch.nn import functional as F
from soft_vocabulary import SoftVocabulary
from graph_soft import GraphSoftVocabulary
from fast_soft import BF16Mixture
from reset_soft import update_logits

CONFIGS=[(f"temperature{temperature}_lr{str(lr).replace('.','p')}",temperature,lr)
 for temperature in [40,80,160] for lr in [.001,.003,.01]]

def reference_tests():
    torch.manual_seed(200032)
    table=F.normalize(torch.randn(19,7,dtype=torch.float64),dim=-1)
    embeddings=torch.randn(19,5,dtype=torch.float64)
    query=torch.randn(3,7,dtype=torch.float64,requires_grad=True)
    downstream=torch.randn(3,5,dtype=torch.float64);temperature=4.
    unit=F.normalize(query,dim=-1)
    probability=torch.softmax(temperature*(unit@table.T),dim=-1)
    objective=((probability@embeddings)*downstream).sum()
    gradient=torch.autograd.grad(objective,query)[0]
    gp=downstream@embeddings.T
    gs=probability.detach()*(gp-(gp*probability.detach()).sum(-1,keepdim=True))
    gu=temperature*(gs@table)
    u=unit.detach()
    manual=(gu-u*(gu*u).sum(-1,keepdim=True))/query.detach().norm(dim=-1,keepdim=True)
    torch.testing.assert_close(gradient,manual,rtol=1e-12,atol=1e-12)
    direction=torch.randn_like(query);epsilon=1e-5
    def value(q):
        p=torch.softmax(temperature*(F.normalize(q,dim=-1)@table.T),dim=-1)
        return ((p@embeddings)*downstream).sum()
    finite=(value(query.detach()+epsilon*direction)-value(query.detach()-epsilon*direction))/(2*epsilon)
    analytic=(gradient*direction).sum()
    torch.testing.assert_close(finite,analytic,rtol=1e-7,atol=1e-8)
    ids=(table@table.T).argmax(-1)
    assert torch.equal(ids,torch.arange(len(table)))
    return {"manual_gradient_max_error":float((gradient-manual).abs().max()),
      "finite_difference_error":float((finite-analytic).abs()),"all_vocabulary_entries_representable":bool(torch.equal(ids,torch.arange(len(table)))),
      "last_entry_representable":int(ids[-1])==len(table)-1,"dtype":"independent CPU float64"}

class QueryVocabulary(SoftVocabulary):
    def __init__(self,prefix,temperature,lr):
        super().__init__(prefix)
        self.bos=prefix.embed_tokens.weight[128000].detach().float().clone().view(1,1,-1)
        self.weight=prefix.embed_tokens.weight.detach()
        self.table=self.metric.table.to(torch.bfloat16)
        self.temperature=temperature;self.lr=lr;self.steps=128;self.tf32=False;self.maximum=128
        self.query=torch.zeros((127,2048),device="cuda",requires_grad=True)
        self.gradient=torch.zeros_like(self.query);self.query.grad=self.gradient
        self.moment=torch.zeros_like(self.query);self.variance=torch.zeros_like(self.query)
        self.h=torch.zeros((128,2048),device="cuda")
        self.tokens=torch.full((128,),128000,device="cuda",dtype=torch.long)
        self.best_tokens=self.tokens.clone();self.position_best_tokens=self.tokens.clone()
        self.best_loss=torch.tensor(float("inf"),device="cuda")
        self.position_best_loss=torch.full((127,),float("inf"),device="cuda")
        self.counter=torch.zeros(1,device="cuda",dtype=torch.long)
        self.loss_trace=torch.zeros(129,device="cuda")
        self.graphs=OrderedDict();self.geometry={};self.capture_events=[]
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        q=F.normalize(self.query[:length-1],dim=-1)
        score=self.temperature*BF16Mixture.apply(q,self.table.T)
        p=F.softmax(score,dim=-1)
        soft=BF16Mixture.apply(p,self.weight)
        predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        loss=error.mean()
        with torch.no_grad():
            ids=score.argmax(-1);self.tokens[1:length].copy_(ids)
            improved=loss.detach()<self.best_loss
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,loss.detach()))
            per=error.detach();better=per<self.position_best_loss[:length-1]
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.minimum(self.position_best_loss[:length-1],per))
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1));self.counter.add_(1)
        return loss
    def train_step(self,length):
        self.gradient.zero_();self.evaluate(length).backward()
        with torch.no_grad():
            update_logits(self.query,self.moment,self.variance,self.gradient,self.lr,1.)
            self.query.copy_(F.normalize(self.query,dim=-1))
    @torch.no_grad()
    def reset(self,h):
        length=len(h);self.h[:length].copy_(h.to("cuda").float())
        self.query.zero_();self.query[:length-1].copy_(F.normalize(self.h[1:length]@self.metric.transform,dim=-1))
        self.gradient.zero_();self.moment.zero_();self.variance.zero_()
        self.tokens.fill_(128000);self.best_tokens.fill_(128000);self.position_best_tokens.fill_(128000)
        self.best_loss.fill_(float("inf"));self.position_best_loss.fill_(float("inf"));self.counter.zero_();self.loss_trace.zero_()
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
        if not torch.isfinite(loss).all():raise RuntimeError("nonfinite query inverse")
        torch.cuda.synchronize();end=time.perf_counter()
        return outputs,{"total_seconds":end-start,"capture_seconds":capture,"input_initialization_seconds":prepared-start-capture,
          "optimization_output_seconds":end-prepared,"loss_trace":loss.tolist(),"whole_sequence_prefix_forwards":129,
          "whole_sequence_prefix_backwards":128,"vocabulary_entries_per_sweep":len(self.weight),
          "full_vocabulary_score_evaluations":129,"full_vocabulary_matrix_products":514,
          "shortlist_size":None,"separate_candidate_verification_calls":0,"model_parameter_updates":0,
          "query_dimensions_per_position":2048,"temperature":self.temperature,"lr":self.lr,
          "parameterization":"unit query, cosine against every prefix-metric embedding; no vocabulary filter",
          "prefix_geometry":"native_length_no_padding"}

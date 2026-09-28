"""Online full-vocabulary input inference; no top-K or separately trained model."""
import time
import torch
from torch.nn import functional as F
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric

class SoftVocabulary:
    def __init__(self,prefix):
        self.prefix=prefix
        self.metric=PrefixWeightMetric(prefix);self.metric.build()
        self.weight=prefix.embed_tokens.weight.detach().float()
        self.bos=self.weight[128000].view(1,1,-1)
        self.transform=self.metric.transform/(self.metric.transform.square().sum().sqrt()/2048**.5)
    def forward(self,z,pe,mask):
        hidden=z.to(torch.bfloat16)
        for layer in self.prefix.layers:
            hidden=self.prefix._hidden(layer(hidden,position_embeddings=pe,attention_mask=mask,use_cache=False))
        return hidden.float()
    def decode(self,observations,scale,lr,objective,steps=128):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter()
        h=observations.to("cuda").float().unsqueeze(0);length=h.shape[1]
        q=F.normalize(h[0,1:]@self.metric.transform,dim=-1)
        logits=(scale*(q@self.metric.table.T)).detach().requires_grad_()
        del q
        optimizer=torch.optim.Adam([logits],lr=lr)
        pos=torch.arange(length,device="cuda").view(1,-1)
        pe=self.prefix.rotary_emb(h.to(torch.bfloat16),pos)
        mask=self.prefix._causal_mask(h.to(torch.bfloat16),start_pos=0,total_tokens=length)
        outputs={};trace=[];best_score=float("inf");best=None
        for iteration in range(steps+1):
            optimizer.zero_grad(set_to_none=True)
            p=F.softmax(logits,dim=-1)
            soft=p@self.weight
            predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
            if objective=="cosine":loss=(1-F.cosine_similarity(predicted[:,1:],h[:,1:],dim=-1)).mean()
            elif objective=="white":loss=((predicted[:,1:]-h[:,1:])@self.transform).square().mean()
            else:raise ValueError(objective)
            value=float(loss.detach())
            ids=torch.cat([torch.tensor([128000],device="cuda"),logits.detach().argmax(-1)])
            if value<best_score:best_score=value;best=ids.clone()
            if iteration in {0,16,32,64,128}:outputs["step"+str(iteration)]=ids.cpu()
            trace.append({"iteration":iteration,"objective":value,
                          "mean_max_probability":float(p.detach().max(-1).values.mean())})
            if not torch.isfinite(loss):raise RuntimeError("nonfinite full-vocabulary inverse")
            if iteration==steps:break
            loss.backward()
            if not torch.isfinite(logits.grad).all():raise RuntimeError("nonfinite full-vocabulary gradient")
            optimizer.step()
        outputs["best_soft_objective"]=best.cpu()
        torch.cuda.synchronize()
        return outputs,{"total_seconds":time.perf_counter()-start,"trace":trace,
          "whole_sequence_prefix_forwards":steps+1,"whole_sequence_prefix_backwards":steps,
          "full_vocabulary_sweeps":steps+1,"vocabulary_entries_per_sweep":len(self.weight),
          "shortlist_size":None,"separate_candidate_verification_calls":0,
          "input_optimization_steps":steps,"model_parameter_updates":0}

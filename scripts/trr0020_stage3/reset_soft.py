"""Full-vocabulary input optimization with reset moments; no candidate pruning."""
import time
import torch
from torch.nn import functional as F
from soft_vocabulary import SoftVocabulary

@torch.no_grad()
def update_logits(logits,moment,variance,gradient,lr,decay):
    moment.lerp_(gradient,.1)
    variance.lerp_(gradient.square(),.005)
    logits.addcdiv_(moment,variance.sqrt().add_(1e-12),value=-lr)
    logits.mul_(decay)

class ResetSoftVocabulary(SoftVocabulary):
    def decode(self,observations,scale,lr,decay,steps=256):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter()
        h=observations.to("cuda").float().unsqueeze(0);length=h.shape[1]
        if scale:
            q=F.normalize(h[0,1:]@self.metric.transform,dim=-1)
            initial=scale*(q@self.metric.table.T)
        else:initial=torch.zeros((length-1,len(self.weight)),device="cuda")
        logits=initial.detach().requires_grad_()
        moment=torch.zeros_like(logits);variance=torch.zeros_like(logits)
        pos=torch.arange(length,device="cuda").view(1,-1)
        pe=self.prefix.rotary_emb(h.to(torch.bfloat16),pos)
        mask=self.prefix._causal_mask(h.to(torch.bfloat16),start_pos=0,total_tokens=length)
        outputs={};trace=[];best_score=float("inf");best=None
        for iteration in range(steps+1):
            logits.grad=None
            probability=F.softmax(logits,dim=-1)
            soft=probability@self.weight
            predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
            loss=(1-F.cosine_similarity(predicted[:,1:],h[:,1:],dim=-1)).mean()
            value=float(loss.detach())
            ids=torch.cat([torch.tensor([128000],device="cuda"),logits.detach().argmax(-1)])
            if value<best_score:best_score=value;best=ids.clone()
            if iteration in {0,16,32,64,128,256}:outputs["step"+str(iteration)]=ids.cpu()
            trace.append({"iteration":iteration,"objective":value,
                "mean_max_probability":float(probability.detach().max(-1).values.mean())})
            if not torch.isfinite(loss):raise RuntimeError("nonfinite full-vocabulary objective")
            if iteration==steps:break
            loss.backward()
            if not torch.isfinite(logits.grad).all():raise RuntimeError("nonfinite gradient")
            update_logits(logits,moment,variance,logits.grad,lr,decay)
            if (iteration+1)%32==0:moment.zero_();variance.zero_()
        outputs["best_soft_objective"]=best.cpu()
        torch.cuda.synchronize()
        return outputs,{"total_seconds":time.perf_counter()-start,"trace":trace,
          "whole_sequence_prefix_forwards":steps+1,"whole_sequence_prefix_backwards":steps,
          "full_vocabulary_sweeps":steps+1,"vocabulary_entries_per_sweep":len(self.weight),
          "shortlist_size":None,"separate_candidate_verification_calls":0,
          "input_optimization_steps":steps,"model_parameter_updates":0,
          "moment_resets":steps//32,"initialization":"uniform" if not scale else "full_vocabulary_metric"}

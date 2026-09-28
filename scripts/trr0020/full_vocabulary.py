"""Direct full-vocabulary approximate A2 updates; no shortlist or verifier."""
import time
import torch
from torch.nn import functional as F
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from token_reconstruction.prefix_fragments import intrinsic_table

class FullVocabularyA2:
    def __init__(self, prefix, descriptor):
        self.prefix=prefix
        self.signature=tuple((id(p),p._version) for p in prefix.parameters())
        metric=PrefixWeightMetric(prefix)
        metric.build()
        self.transform=metric.transform
        del metric.table,metric
        self.descriptor=descriptor
        with torch.inference_mode():
            if descriptor=="embedding":
                raw=prefix.embed_tokens.weight.detach().float()
            elif descriptor=="intrinsic":
                raw=intrinsic_table(prefix,256).float()
            elif descriptor=="bos":
                raw=torch.empty(prefix.embed_tokens.weight.shape,device="cuda",dtype=torch.float32)
                for lo in range(0,len(raw),256):
                    ids=torch.arange(lo,min(lo+256,len(raw)),device="cuda")
                    pair=torch.stack([torch.full_like(ids,128000),ids],dim=-1)
                    raw[lo:lo+len(ids)]=prefix.forward_full(pair)[:,1].float()
            else:raise ValueError(descriptor)
            self.raw=raw
            self.norms=raw.square().sum(-1)
            self.table=torch.empty_like(raw)
            for lo in range(0,len(raw),256):
                self.table[lo:lo+256]=F.normalize(raw[lo:lo+256]@self.transform,dim=-1)
        self.vocab=len(raw)
    def check(self):
        if self.signature!=tuple((id(p),p._version) for p in self.prefix.parameters()):
            raise RuntimeError("prefix changed: rebuild deterministic descriptors")
    @torch.inference_mode()
    def white_select(self,q):
        q=F.normalize(q@self.transform,dim=-1)
        return (q@self.table.T).argmax(-1)
    @torch.inference_mode()
    def shifted_select(self,h,context):
        # Exact cosine for the APPROXIMATE activation D(v)+context.
        numerator=h@self.raw.T+(h*context).sum(-1,keepdim=True)
        denominator=(self.norms[None,:]+2*context@self.raw.T+
                     context.square().sum(-1,keepdim=True)).clamp_min(1e-20).sqrt()
        return (numerator/denominator).argmax(-1)
    @torch.inference_mode()
    def decode(self,observations,rule,step=1.0,iterations=32):
        self.check();torch.cuda.synchronize();start=time.perf_counter()
        h=observations.to("cuda").float()
        ids=self.white_select(h);ids[0]=128000
        outputs={};trace=[];snapshots={0,1,2,4,8,16,32}
        best_score=float("inf");best_ids=None
        per_best=torch.full((len(h),),float("inf"),device="cuda")
        per_ids=ids.clone()
        sweeps=1
        for iteration in range(iterations+1):
            prediction=self.prefix.forward_full(ids[None])[0].float()
            error=1-F.cosine_similarity(prediction,h,dim=-1)
            score=float(error[1:].mean())
            if not torch.isfinite(error).all():raise RuntimeError("nonfinite approximation")
            if score<best_score:best_score=score;best_ids=ids.clone()
            improved=error<per_best
            per_ids=torch.where(improved,ids,per_ids)
            per_best=torch.minimum(per_best,error)
            if iteration in snapshots:outputs["step"+str(iteration)]=ids.cpu()
            trace.append({"iteration":iteration,"mean_cosine_error":score,
                          "maximum_cosine_error":float(error[1:].max())})
            if iteration==iterations:break
            if rule=="white":
                proposed=self.white_select(self.raw[ids]+step*(h-prediction))
            elif rule=="shifted":
                proposed=self.shifted_select(h,prediction-self.raw[ids])
            else:raise ValueError(rule)
            proposed[0]=128000
            trace[-1]["changed_tokens"]=int((proposed[1:]!=ids[1:]).sum())
            ids=proposed;sweeps+=1
        outputs["best_sequence"]=best_ids.cpu()
        outputs["best_positions"]=per_ids.cpu()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        return outputs,{"total_seconds":elapsed,"trace":trace,
           "whole_sequence_prefix_forwards":iterations+1,"full_vocabulary_sweeps":sweeps,
           "vocabulary_entries_per_sweep":self.vocab,"shortlist_size":None,
           "separate_candidate_verification_calls":0,"online_training_steps":0}

"""Soft reconstruction followed by direct all-vocabulary quadratic updates."""
from collections import OrderedDict
import time
import torch
from torch.nn import functional as F
from discrete_soft import DiscreteSoftVocabulary
from proximal import ProximalVocabulary
from discrete_parallel import DiscreteParallel

CONFIGS=[(f"warm{warm}_{metric}_beta{str(beta).replace('.','p')}",warm,metric,beta)
         for warm in [64,128] for metric in ["raw","white"] for beta in [.01,.03,.1]]
SNAPSHOTS={0,1,2,4,8,16,32}

class WarmProximal(ProximalVocabulary):
    def __init__(self,soft,metric,beta):
        # Share immutable public weights and weight-derived tables; do not rebuild them.
        self.prefix=soft.prefix;self.metric=soft.metric;self.transform=soft.transform
        self.bos=soft.bos;self.weight=soft.weight
        self.kind=metric;self.beta=beta;self.steps=32
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
        self.trace=torch.zeros(33,device="cuda");self.error=torch.zeros(127,device="cuda")
        self.graphs=OrderedDict();self.geometry={};self.capture_events=[]
    @torch.no_grad()
    def reset(self,h,initial=None):
        length=len(h);self.h[:length].copy_(h.to("cuda").float())
        self.tokens.fill_(128000)
        if initial is not None:
            if initial.shape!=(length,) or initial.dtype!=torch.long:raise ValueError("invalid initial sequence")
            if int(initial[0])!=128000:raise ValueError("missing BOS")
            self.tokens[:length].copy_(initial.to("cuda"))
        self.z.zero_();self.z[:length-1].copy_(self.weight[self.tokens[1:length]].float())
        self.gradient.zero_();self.best_tokens.fill_(128000);self.position_tokens.fill_(128000)
        self.best_loss.fill_(float("inf"));self.position_best.fill_(float("inf"))
        self.counter.zero_();self.trace.zero_();self.error.zero_()
    def decode(self,h,initial):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter();length=len(h)
        capture=self.ensure(length);self.reset(h,initial);torch.cuda.synchronize();prepared=time.perf_counter()
        train,evaluate=self.graphs[length];outputs={}
        for i in range(self.steps):
            if i in SNAPSHOTS:outputs["step"+str(i)]=self.tokens[:length].cpu()
            train.replay()
        evaluate.replay()
        outputs["step32"]=self.tokens[:length].cpu()
        outputs["best_objective"]=self.best_tokens[:length].cpu()
        outputs["best_position_error"]=self.position_tokens[:length].cpu()
        losses=self.trace.cpu()
        if not torch.isfinite(losses).all():raise RuntimeError("nonfinite direct objective")
        torch.cuda.synchronize();end=time.perf_counter()
        return outputs,{"total_seconds":end-start,"capture_seconds":capture,
          "input_initialization_seconds":prepared-start-capture,"optimization_output_seconds":end-prepared,
          "loss_trace":losses.tolist(),"whole_sequence_prefix_forwards":33,"whole_sequence_prefix_backwards":32,
          "full_vocabulary_sweeps":32,"vocabulary_entries_per_sweep":len(self.weight),"shortlist_size":None,
          "separate_candidate_verification_calls":0,"input_optimization_steps":32,"model_parameter_updates":0,
          "metric":self.kind,"beta":self.beta,"score_numerics":"BF16_QUERY_AND_WEIGHTS_FP32_OUTPUT",
          "decision":"direct full-vocabulary quadratic argmax from the current whole hard sequence"}

class WarmRefinement:
    def __init__(self,prefix,warm,metric,beta):
        self.soft=DiscreteSoftVocabulary(prefix,"gini003",.003,0.,32);self.soft.steps=warm
        self.direct=WarmProximal(self.soft,metric,beta);self.warm=warm
        assert self.soft.metric is self.direct.metric and self.soft.weight is self.direct.weight
    @property
    def capture_events(self):
        return {"soft":self.soft.capture_events,"direct":self.direct.capture_events}
    def decode(self,h):
        torch.cuda.synchronize();start=time.perf_counter()
        warm,ss=self.soft.decode(h)
        direct,ds=self.direct.decode(h,warm["step"+str(self.warm)])
        outputs={"warm_"+key:value for key,value in warm.items()};outputs.update(direct)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        return outputs,{"total_seconds":elapsed,"soft":ss,"direct":ds,
          "capture_seconds":ss["capture_seconds"]+ds["capture_seconds"],
          "whole_sequence_prefix_forwards":self.warm+34,"whole_sequence_prefix_backwards":self.warm+32,
          "full_vocabulary_sweeps":ss["full_vocabulary_sweeps"]+32,
          "vocabulary_entries_per_sweep":128256,"shortlist_size":None,"separate_candidate_verification_calls":0,
          "model_parameter_updates":0,"shared_prefix_metric_and_dictionary":True,
          "inference_boundary":"both stages including capture, initialization, synchronization and output transfer"}

def eager_reference(engine,h,initial):
    """Independent eager control flow for the direct stage; same declared prefix arithmetic."""
    length=len(h);pe,mask=engine.geometry[length]
    target=h.to("cuda").float()[None,1:]
    ids=initial.to("cuda").clone()
    z=engine.weight[ids[1:]].float().detach().requires_grad_()
    best=ids.clone();position=ids.clone()
    best_loss=float("inf");position_loss=torch.full((length-1,),float("inf"),device="cuda")
    outputs={};trace=[]
    for step in range(33):
        if step in SNAPSHOTS:outputs["step"+str(step)]=ids.cpu()
        full=torch.cat([engine.bos,z[None]],dim=1)
        predicted=DiscreteParallel.diagonal_forward(engine,full,pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],target,dim=-1))[0]
        value=float(error.detach().mean());trace.append(value)
        if value<best_loss:best=ids.clone();best_loss=value
        with torch.no_grad():
            better=error<position_loss
            position[1:]=torch.where(better,ids[1:],position[1:])
            position_loss=torch.minimum(position_loss,error)
        if step==32:break
        g=torch.autograd.grad(error.sum(),z)[0]
        with torch.no_grad():
            if engine.kind=="white":
                dual=g@engine.rinv.T;metric_z=(z@engine.r)@engine.r.T
            else:dual=g;metric_z=z
            curvature=(engine.beta*dual.square().sum(-1)/(2*error.clamp_min(1e-8))).clamp(1e-6,1e6)
            query=curvature[:,None]*metric_z-g
            scores=torch.mm(query.to(torch.bfloat16),engine.weight.T,out_dtype=torch.float32)-.5*curvature[:,None]*engine.norm
            ids[1:]=scores.argmax(-1)
            z.copy_(engine.weight[ids[1:]].float())
    outputs["best_objective"]=best.cpu();outputs["best_position_error"]=position.cpu()
    return outputs,trace

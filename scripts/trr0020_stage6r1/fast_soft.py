"""Approximate full-vocabulary mixture arithmetic and fused online-input updates."""
import time
import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from graph_soft import GraphSoftVocabulary

class BF16Mixture(torch.autograd.Function):
    @staticmethod
    def forward(ctx,probability,weight):
        ctx.save_for_backward(weight)
        return torch.mm(probability.to(torch.bfloat16),weight,out_dtype=torch.float32)
    @staticmethod
    def backward(ctx,gradient):
        (weight,)=ctx.saved_tensors
        return torch.mm(gradient.to(torch.bfloat16),weight.T,out_dtype=torch.float32),None

@triton.jit
def update_kernel(Z,M,V,G,E,LR,DECAY,N:tl.constexpr,VOCAB:tl.constexpr,ADAPTIVE:tl.constexpr,BLOCK:tl.constexpr):
    ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=ix<N
    z=tl.load(Z+ix,mask,0);m=tl.load(M+ix,mask,0)
    v=tl.load(V+ix,mask,0);g=tl.load(G+ix,mask,0)
    lr=tl.load(LR);decay=tl.load(DECAY)
    if ADAPTIVE:
        error=tl.load(E+ix//VOCAB,mask,0)
        lr=lr*tl.sqrt(tl.minimum(tl.maximum(error/.05,.01),1.))
    m=m+.1*(g-m);v=v+.005*(g*g-v)
    z=(z-lr*tl.div_rn(m,tl.sqrt(v)+1e-12))*decay
    tl.store(Z+ix,z,mask);tl.store(M+ix,m,mask);tl.store(V+ix,v,mask)

def fused_update(z,m,v,g,error,lr,decay,n,vocab,adaptive):
    update_kernel[(triton.cdiv(n,1024),)](z,m,v,g,error,lr,decay,n,vocab,adaptive,1024,num_warps=4,enable_fp_fusion=False)

class FastSoftVocabulary(GraphSoftVocabulary):
    def __init__(self,prefix,mode="constant",steps=256):
        super().__init__(prefix,"cosine",.6,False)
        self.bos=prefix.embed_tokens.weight[128000].detach().float().clone().view(1,1,-1)
        self.weight=prefix.embed_tokens.weight.detach()
        self.mode=mode;self.steps=steps
        self.lr_tensor=torch.tensor(.6,device="cuda")
        self.decay_tensor=torch.tensor(.98,device="cuda")
        self.position_error=torch.zeros(127,device="cuda")
        self.position_best_loss=torch.full((127,),float("inf"),device="cuda")
        self.position_best_tokens=torch.full((128,),128000,device="cuda",dtype=torch.long)
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        probability=F.softmax(self.logits[:length-1],dim=-1)
        soft=BF16Mixture.apply(probability,self.weight)
        predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        loss=error.mean()
        with torch.no_grad():
            ids=self.logits[:length-1].argmax(-1)
            self.tokens[1:length].copy_(ids)
            improved=loss.detach()<self.best_loss
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,loss.detach()))
            per=error.detach()
            better=per<self.position_best_loss[:length-1]
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.minimum(self.position_best_loss[:length-1],per))
            self.position_error[:length-1].copy_(per)
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.counter.add_(1)
        return loss
    def train_step(self,length):
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        fused_update(self.logits,self.moment,self.variance,self.gradient,self.position_error,
                     self.lr_tensor,self.decay_tensor,(length-1)*self.vocab,self.vocab,self.mode=="adaptive")
    @torch.no_grad()
    def reset(self,h):
        super().reset(h)
        self.position_best_loss.fill_(float("inf"));self.position_best_tokens.fill_(128000)
        self.position_error.zero_();self.lr_tensor.fill_(.6);self.decay_tensor.fill_(.98)
    def decode(self,observations):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter()
        length=len(observations);capture=self.ensure(length)
        self.reset(observations);torch.cuda.synchronize();prepared=time.perf_counter()
        training,evaluating=self.graphs[length];outputs={}
        for iteration in range(self.steps):
            if self.mode=="cool":
                if iteration==self.steps//4:
                    self.lr_tensor.fill_(.3);self.decay_tensor.fill_(.99)
                if iteration==self.steps//2:
                    self.lr_tensor.fill_(.1);self.decay_tensor.fill_(.999)
            training.replay()
            if iteration in {0,16,32,64,128,256}:outputs["step"+str(iteration)]=self.tokens[:length].cpu()
            if (iteration+1)%32==0:self.moment.zero_();self.variance.zero_()
        evaluating.replay()
        outputs["step"+str(self.steps)]=self.tokens[:length].cpu()
        outputs["best_soft_objective"]=self.best_tokens[:length].cpu()
        outputs["best_position_soft_error"]=self.position_best_tokens[:length].cpu()
        losses=self.loss_trace[:self.steps+1].cpu()
        if not torch.isfinite(losses).all():raise RuntimeError("nonfinite fast inverse")
        torch.cuda.synchronize();end=time.perf_counter()
        return outputs,{"total_seconds":end-start,"capture_seconds":capture,
          "input_initialization_seconds":prepared-start-capture,"optimization_output_seconds":end-prepared,
          "loss_trace":losses.tolist(),"whole_sequence_prefix_forwards":self.steps+1,
          "whole_sequence_prefix_backwards":self.steps,"full_vocabulary_sweeps":self.steps+1,
          "vocabulary_entries_per_sweep":self.vocab,"shortlist_size":None,
          "separate_candidate_verification_calls":0,"input_optimization_steps":self.steps,
          "model_parameter_updates":0,"mixture_numerics":"BF16_OPERANDS_FP32_OUTPUT",
          "optimizer_numerics":"FUSED_FP32_NO_FMA","initialization_numerics":"FP32_TF32_DISABLED",
          "prefix_geometry":"native_length_no_padding","mode":self.mode}

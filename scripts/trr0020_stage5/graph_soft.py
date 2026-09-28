"""Full-vocabulary input optimization, captured without per-iteration CPU dispatch."""
from collections import OrderedDict
import gc,time
import torch
from torch.nn import functional as F
from soft_vocabulary import SoftVocabulary
from reset_soft import update_logits

class GraphSoftVocabulary(SoftVocabulary):
    def __init__(self,prefix,objective="cosine",lr=.6,tf32=False):
        torch.backends.cuda.matmul.allow_tf32=False
        super().__init__(prefix)
        self.objective=objective;self.lr=lr;self.tf32=tf32
        self.maximum=128;self.vocab=len(self.weight)
        self.logits=torch.zeros((127,self.vocab),device="cuda",requires_grad=True)
        self.gradient=torch.zeros_like(self.logits);self.logits.grad=self.gradient
        self.moment=torch.zeros_like(self.logits);self.variance=torch.zeros_like(self.logits)
        self.h=torch.zeros((128,2048),device="cuda")
        self.tokens=torch.full((128,),128000,device="cuda",dtype=torch.long)
        self.best_tokens=self.tokens.clone()
        self.best_loss=torch.tensor(float("inf"),device="cuda")
        self.loss_trace=torch.zeros(1025,device="cuda")
        self.counter=torch.zeros(1,device="cuda",dtype=torch.long)
        self.graphs=OrderedDict();self.geometry={}
        self.capture_events=[]
    def loss_value(self,predicted,h):
        if self.objective=="cosine":
            return (1-F.cosine_similarity(predicted,h,dim=-1)).mean()
        if self.objective=="white":
            return ((predicted-h)@self.transform).square().mean()
        mse=((predicted-h).square().mean(-1)/h.square().mean(-1).clamp_min(1e-8)).mean()
        if self.objective=="mse":return mse
        if self.objective=="mixed":return (1-F.cosine_similarity(predicted,h,dim=-1)).mean()+.1*mse
        raise ValueError(self.objective)
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        probability=F.softmax(self.logits[:length-1],dim=-1)
        soft=probability@self.weight
        predicted=self.forward(torch.cat([self.bos,soft[None]],dim=1),pe,mask)
        loss=self.loss_value(predicted[:,1:],self.h[None,1:length])
        with torch.no_grad():
            ids=self.logits[:length-1].argmax(-1)
            self.tokens[1:length].copy_(ids)
            improved=loss.detach()<self.best_loss
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.minimum(self.best_loss,loss.detach()))
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1))
            self.counter.add_(1)
        return loss
    def train_step(self,length):
        self.gradient.zero_()
        loss=self.evaluate(length);loss.backward()
        update_logits(self.logits,self.moment,self.variance,self.gradient,self.lr,.98)
    @torch.no_grad()
    def reset(self,h):
        length=len(h)
        torch.backends.cuda.matmul.allow_tf32=False
        self.h[:length].copy_(h.to("cuda").float())
        q=F.normalize(self.h[1:length]@self.metric.transform,dim=-1)
        self.logits.zero_();self.logits[:length-1].copy_(80*(q@self.metric.table.T))
        self.gradient.zero_();self.moment.zero_();self.variance.zero_()
        self.best_loss.fill_(float("inf"));self.counter.zero_();self.loss_trace.zero_()
        self.tokens.fill_(128000);self.best_tokens.fill_(128000)
    def ensure(self,length):
        if length in self.graphs:
            self.graphs.move_to_end(length);return 0.
        if not 2<=length<=128:raise ValueError("qualified geometry2..128")
        torch.cuda.synchronize();start=time.perf_counter()
        if len(self.graphs)>=2:
            old,pair=self.graphs.popitem(last=False)
            del pair;self.geometry.pop(old);gc.collect();torch.cuda.empty_cache()
        with torch.no_grad():
            dummy=self.prefix.forward_full(torch.full((1,length),128000,device="cuda",dtype=torch.long))[0].float()
            pos=torch.arange(length,device="cuda").view(1,-1)
            pe=self.prefix.rotary_emb(dummy[None].to(torch.bfloat16),pos)
            mask=self.prefix._causal_mask(dummy[None].to(torch.bfloat16),start_pos=0,total_tokens=length)
            self.geometry[length]=(pe,mask)
            self.reset(dummy)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        torch.backends.cuda.matmul.allow_tf32=self.tf32
        with torch.cuda.stream(stream):
            for _ in range(3):self.train_step(length)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        training=torch.cuda.CUDAGraph()
        with torch.cuda.graph(training,stream=stream):
            self.train_step(length)
        evaluating=torch.cuda.CUDAGraph()
        with torch.cuda.graph(evaluating,stream=stream):
            with torch.no_grad():self.evaluate(length)
        torch.cuda.synchronize();torch.backends.cuda.matmul.allow_tf32=False
        self.graphs[length]=(training,evaluating)
        elapsed=time.perf_counter()-start
        self.capture_events.append({"length":length,"seconds":elapsed,
            "reserved_bytes":torch.cuda.memory_reserved(),"cached_lengths":list(self.graphs)})
        return elapsed
    def decode(self,observations,steps=256):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter()
        length=len(observations);capture=self.ensure(length)
        self.reset(observations);torch.cuda.synchronize();prepared=time.perf_counter()
        training,evaluating=self.graphs[length];outputs={}
        for iteration in range(steps):
            training.replay()
            if iteration in {0,16,32,64,128,256}:
                outputs["step"+str(iteration)]=self.tokens[:length].cpu()
            if (iteration+1)%32==0:self.moment.zero_();self.variance.zero_()
        evaluating.replay()
        outputs["step"+str(steps)]=self.tokens[:length].cpu()
        outputs["best_soft_objective"]=self.best_tokens[:length].cpu()
        losses=self.loss_trace[:steps+1].cpu()
        if not torch.isfinite(losses).all():raise RuntimeError("nonfinite graph inverse")
        torch.cuda.synchronize();end=time.perf_counter()
        return outputs,{"total_seconds":end-start,"capture_seconds":capture,
          "input_initialization_seconds":prepared-start-capture,"optimization_output_seconds":end-prepared,
          "loss_trace":losses.tolist(),"whole_sequence_prefix_forwards":steps+1,
          "whole_sequence_prefix_backwards":steps,"full_vocabulary_sweeps":steps+1,
          "vocabulary_entries_per_sweep":self.vocab,"shortlist_size":None,
          "separate_candidate_verification_calls":0,"input_optimization_steps":steps,
          "model_parameter_updates":0,"tf32_optimization":self.tf32,
          "initialization_numerics":"FP32_TF32_DISABLED","prefix_geometry":"native_length_no_padding"}

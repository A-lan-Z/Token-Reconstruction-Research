"""Native-context replay engine for causal whole-vocabulary input probabilities."""
from pathlib import Path
import sys,time,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage27"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage32"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage33"))
from qualify_gpu import n,F,public_parameters,forward,vjp,commit
from probability_gradient import gradient
from budget_step import update
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric

class CurrentVocabulary:
    def __init__(self,prefix):
        self.prefix=prefix;self.layers=public_parameters(prefix);self.E=prefix.embed_tokens.weight.detach()
        self.metric=PrefixWeightMetric(prefix);self.metric.build();self.vocab,self.width=self.E.shape
        self.past=[(torch.zeros(p["kv_heads"],128,p["head_dim"],device="cuda"),torch.zeros(p["kv_heads"],128,p["head_dim"],device="cuda")) for p in self.layers]
        self.h=torch.zeros(1,self.width,device="cuda")
        hd=self.layers[0]["head_dim"];self.co=torch.ones(1,hd,device="cuda");self.si=torch.zeros(1,hd,device="cuda")
        self.z=torch.zeros(1,self.vocab,device="cuda");self.factor=torch.tensor(2.,device="cuda")
        self.counter=torch.zeros(1,device="cuda",dtype=torch.long)
        self.trace=torch.zeros(8,5,device="cuda");self.logit_trace=torch.zeros(9,1,self.vocab,device="cuda")
        self.final_mean=torch.zeros(1,self.width,device="cuda");self.final_output=torch.zeros_like(self.final_mean)
        self.final_error=torch.zeros(1,device="cuda");self.confidence=torch.zeros(1,device="cuda")
        self.token=torch.zeros(1,dtype=torch.long,device="cuda");self.commit_output=torch.zeros_like(self.final_mean)
        self.current=[(torch.zeros(p["kv_heads"],1,p["head_dim"],device="cuda"),torch.zeros(p["kv_heads"],1,p["head_dim"],device="cuda")) for p in self.layers]
        self.stream=torch.cuda.Stream();self.graphs={};self.capture_events=[];self.init_graph=None
    def history(self,pos):return [(k[:,:pos],v[:,:pos]) for k,v in self.past]
    @torch.no_grad()
    def initialize(self):
        query=F.normalize(self.h@self.metric.transform,dim=-1)
        self.z.copy_(80*(query@self.metric.table.T))
        self.counter.zero_();self.trace.zero_();self.logit_trace.zero_();self.logit_trace[0].copy_(self.z)
    @torch.no_grad()
    def train(self,pos):
        p=self.z.softmax(-1);mean=p@self.E
        out,caches,_=forward(mean,self.layers,self.history(pos),self.co,self.si)
        G,info=gradient(out,self.h,self.E,lambda d:vjp(d,self.layers,caches,self.co,self.si))
        new,step=update(self.z,G,info["loss"],self.factor,4)
        record=torch.stack([info["loss"][0],p.amax(),step["requested_budget"][0],step["formula_kl"][0],step["budget_used"][0]])
        self.trace.index_copy_(0,self.counter,record[None])
        self.z.copy_(new);self.counter.add_(1);self.logit_trace.index_copy_(0,self.counter,self.z[None])
    @torch.no_grad()
    def evaluate(self,pos):
        p=self.z.softmax(-1);mean=p@self.E
        out,_,_=forward(mean,self.layers,self.history(pos),self.co,self.si)
        error=1-(F.normalize(out,dim=-1)*F.normalize(self.h,dim=-1)).sum(-1)
        self.final_mean.copy_(mean);self.final_output.copy_(out);self.final_error.copy_(error)
        self.confidence.copy_(p.amax(-1));self.token.copy_(self.z.argmax(-1))
    @torch.no_grad()
    def commit_token(self,pos):
        output,_,current=forward(self.E.index_select(0,self.token),self.layers,self.history(pos),self.co,self.si)
        self.commit_output.copy_(output)
        for (pk,pv),(ck,cv),(ok,ov) in zip(self.past,current,self.current):
            ok.copy_(ck);ov.copy_(cv);pk[:,pos:pos+1].copy_(ck);pv[:,pos:pos+1].copy_(cv)
    @torch.no_grad()
    def ensure(self,pos):
        if pos in self.graphs:return 0.
        n.sync();start=time.perf_counter();stream=self.stream;stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.initialize();self.train(pos);self.evaluate(pos);self.commit_token(pos)
        torch.cuda.current_stream().wait_stream(stream);n.sync()
        if self.init_graph is None:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):self.initialize()
            self.init_graph=graph
        training=torch.cuda.CUDAGraph()
        self.initialize();n.sync()
        with torch.cuda.graph(training,stream=stream):self.train(pos)
        evaluating=torch.cuda.CUDAGraph()
        with torch.cuda.graph(evaluating,stream=stream):self.evaluate(pos)
        committing=torch.cuda.CUDAGraph()
        with torch.cuda.graph(committing,stream=stream):self.commit_token(pos)
        n.sync();self.graphs[pos]=(training,evaluating,committing)
        seconds=time.perf_counter()-start
        self.capture_events.append({"position":pos,"seconds":seconds,"peak_reserved":torch.cuda.max_memory_reserved(),"reserved":torch.cuda.memory_reserved()})
        return seconds
    @torch.no_grad()
    def set_context(self,target,past,co,si,factor):
        self.h.copy_(target);self.co.copy_(co);self.si.copy_(si);self.factor.fill_(factor)
        for (pk,pv),(k,v) in zip(self.past,past):
            pk.zero_();pv.zero_();pk[:,:k.shape[1]].copy_(k);pv[:,:v.shape[1]].copy_(v)
    @torch.no_grad()
    def run_current(self,target,past,co,si,factor,steps,replay=True):
        if not 1<=steps<=8:raise ValueError("steps outside qualified geometry")
        preparation_start=time.perf_counter()
        pos=past[0][0].shape[1];self.set_context(target,past,co,si,factor)
        capture=self.ensure(pos);self.set_context(target,past,co,si,factor)
        n.sync();start=time.perf_counter();context_seconds=start-preparation_start
        initialization_start=torch.cuda.Event(enable_timing=True);initialization_start.record()
        if replay:self.init_graph.replay()
        else:self.initialize()
        initialize_end=torch.cuda.Event(enable_timing=True);initialize_end.record()
        training,evaluating,committing=self.graphs[pos]
        for _ in range(steps):
            if replay:training.replay()
            else:self.train(pos)
        if replay:evaluating.replay()
        else:self.evaluate(pos)
        n.sync();inference=time.perf_counter()-start
        start=time.perf_counter()
        if replay:committing.replay()
        else:self.commit_token(pos)
        n.sync();commit_seconds=time.perf_counter()-start
        start=time.perf_counter();actual=[]
        saved=self.logit_trace[:steps+1].cpu()
        for old,new in zip(saved[:-1],saved[1:]):
            a=old.double().log_softmax(-1);b=new.double().log_softmax(-1)
            actual.append((a.exp()*(a-b)).sum(-1))
        direct=torch.stack(actual)[:,0]
        data={"logits":self.z.cpu(),"mixture":self.final_mean.cpu(),"prefix_output":self.final_output.cpu(),
          "observed_error":self.final_error.cpu(),"confidence":self.confidence.cpu(),"token":self.token.cpu(),
          "trace":self.trace[:steps].cpu(),"direct_kl":direct,"requested_budget":self.trace[:steps,2].cpu().double(),
          "commit_output":self.commit_output.cpu()}
        for i,(key,value) in enumerate(self.current):data[f"commit_key_{i}"]=key.cpu();data[f"commit_value_{i}"]=value.cpu()
        numeric=all(bool(torch.isfinite(v).all()) for v in data.values())
        passed=numeric and bool((direct<=data["requested_budget"]+2e-5).all()) and bool((direct>=-2e-5).all())
        return data,{"capture_seconds":capture,"inference_seconds":inference,"emitted_token_commit_seconds":commit_seconds,
          "validation_transfer_seconds":time.perf_counter()-start,"numeric_valid":numeric,"passed":passed,
          "prefix_forwards":steps+1,"prefix_vjps":steps,"commit_forwards":1,"initialization_included":True,"initialization_gpu_seconds":initialization_start.elapsed_time(initialize_end)/1000,
          "context_and_capture_seconds":context_seconds,
          "history_transfer_excluded":True,"replay":replay,"timing_scope":"initialization+current-token updates+final soft evaluation; history transfer, capture, commit, and validation separately disclosed"}

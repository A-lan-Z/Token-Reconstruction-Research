"""Equivalent replay capture using one owned stream per optimizer stage."""
from collections import OrderedDict
from types import MethodType
import gc,time
import torch
from warm_refinement import WarmRefinement

def ensure_owned_stream(self,length):
    if length in self.graphs:self.graphs.move_to_end(length);return 0.
    if not 2<=length<=128:raise ValueError("qualified geometry2..128")
    torch.cuda.synchronize();start=time.perf_counter()
    if len(self.graphs)>=2:
        old,pair=self.graphs.popitem(last=False);del pair
        self.geometry.pop(old);gc.collect();torch.cuda.empty_cache()
    with torch.no_grad():
        dummy=self.prefix.forward_full(torch.full((1,length),128000,device="cuda",dtype=torch.long))[0].float()
        pe=self.prefix.rotary_emb(dummy[None].to(torch.bfloat16),torch.arange(length,device="cuda").view(1,-1))
        mask=self.prefix._causal_mask(dummy[None].to(torch.bfloat16),start_pos=0,total_tokens=length)
        self.geometry[length]=(pe,mask);self.reset(dummy)
    stream=self.capture_stream;stream.wait_stream(torch.cuda.current_stream())
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

class CanonicalRefinement(WarmRefinement):
    def __init__(self,prefix):
        super().__init__(prefix,128,"white",.03)
        for engine in [self.soft,self.direct]:
            engine.capture_stream=torch.cuda.Stream()
            engine.ensure=MethodType(ensure_owned_stream,engine)

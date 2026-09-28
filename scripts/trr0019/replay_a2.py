"""Fixed-geometry GPU replay for the native K256 A2 decision rule.

Each position retains its native attention length. Graphs share temporary
storage and run causally in capture order. Only inputs and outputs persist
across records; every replay starts by rebuilding the BOS context.
"""
import time
import torch
from torch.nn import functional as F
from shared_context import shared_candidates
from shared_attention import fast_candidates

class ReplayA2:
    @torch.inference_mode()
    def __init__(self,prefix,max_positions=128,fast=False):
        if max_positions<2 or max_positions>128:raise ValueError('qualified length2..128')
        self.prefix=prefix;self.maximum=max_positions;self.fast=fast
        self.candidates=torch.full((max_positions,256),128000,dtype=torch.long,device='cuda')
        self.observations=torch.zeros((max_positions,2048),dtype=torch.float32,device='cuda')
        self.tokens=torch.full((max_positions,),128000,dtype=torch.long,device='cuda')
        self.scores=torch.empty((max_positions-1,256),dtype=torch.float32,device='cuda')
        self.mse=torch.empty_like(self.scores)
        self.bos=torch.tensor([[128000]],device='cuda')
        self.signature=self.layout_signature();self.graphs=[]
        candidate_call=fast_candidates if fast else shared_candidates
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        started=time.perf_counter()
        with torch.cuda.stream(stream):
            warm=prefix.new_cache();prefix.run_cached(self.bos,warm,0)
            for pos in range(1,max_positions):
                response=candidate_call(prefix,warm,self.candidates[pos],pos)
                prefix.run_cached(self.bos,warm,pos)
            del warm,response
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        warmed=time.perf_counter();pool=torch.cuda.graph_pool_handle();cache=prefix.new_cache()
        for pos in range(max_positions):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=pool,stream=stream):
                if pos==0:
                    prefix.run_cached(self.bos,cache,0)
                    self.tokens[0].fill_(128000)
                else:
                    response=candidate_call(prefix,cache,self.candidates[pos],pos).float()
                    score=F.cosine_similarity(response,self.observations[pos:pos+1],dim=-1)
                    mse=(response-self.observations[pos]).square().mean(-1)
                    winner=self.candidates[pos].gather(0,score.argmax().reshape(1))
                    self.tokens[pos].copy_(winner.reshape(()))
                    self.scores[pos-1].copy_(score);self.mse[pos-1].copy_(mse)
                    prefix.run_cached(winner.reshape(1,1),cache,pos)
            self.graphs.append(graph)
            if pos:del response,score,mse,winner
        self.last_capture_cache=cache
        torch.cuda.synchronize()
        self.setup={'warmup_seconds':warmed-started,'capture_seconds':time.perf_counter()-warmed,
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
    def layout_signature(self):
        return tuple((name,id(t),t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device))
                     for name,t in list(self.prefix.named_parameters())+list(self.prefix.named_buffers()))
    def check(self):
        if self.layout_signature()!=self.signature:raise RuntimeError('prefix tensor storage/layout changed; recapture required')
    @torch.inference_mode()
    def run(self,h,ids):
        self.check();length=len(h)
        if not 2<=length<=self.maximum or tuple(h.shape)!=(length,2048) or tuple(ids.shape)!=(length,256):
            raise ValueError('unsupported replay input geometry')
        if ids.dtype!=torch.int64:raise ValueError('token IDs must be int64')
        self.candidates[:length].copy_(ids);self.observations[:length].copy_(h)
        for graph in self.graphs[:length]:graph.replay()
        return {'tokens':self.tokens[:length],'candidates':self.candidates[:length],
                'scores':self.scores[:length-1],'mse':self.mse[:length-1]}

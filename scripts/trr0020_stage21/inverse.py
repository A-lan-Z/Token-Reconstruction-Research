"""Full-vocabulary readout after matrix-free local raw-activation inversion."""
from collections import OrderedDict
import gc,time
import torch
from torch.nn import functional as F
from mean_readout import MeanReadout,READOUTS
from fast_soft import BF16Mixture
from linearized_prefix import forward,diagonal_jvp,public_parameters
from krylov_graph import direction
CHECKPOINTS=(4,8,16)
CONFIGS=[(f"warm{warm}_k{k}_damp{str(damp).replace('.','p')}",warm,k,damp)
         for warm in [0,32] for k in [8,16] for damp in [.5,1.]]
class LocalInverse(MeanReadout):
    def __init__(self,prefix,warm,k,damping):
        super().__init__(prefix,warm)
        self.prefix=prefix;self.k=k;self.damping=damping;self.layers=public_parameters(prefix)
        self.radius=self.raw_norm.sqrt().median()
        self.x=torch.zeros((128,2048),device="cuda");self.target=torch.zeros_like(self.x)
        self.counter=torch.zeros(1,device="cuda",dtype=torch.long)
        self.trace=torch.zeros((16,5),device="cuda")
        self.solver_info=torch.zeros(128,device="cuda",dtype=torch.int32)
        self.local_graphs=OrderedDict();self.local_geometry={};self.local_capture_events=[]
        self.local_stream=torch.cuda.Stream()
    @torch.no_grad()
    def reset_local(self,mean,observations):
        length=len(observations)
        self.x[0].copy_(self.weight[128000]);self.x[1:length].copy_(mean)
        self.target[:length].copy_(observations.to("cuda").float())
        self.counter.zero_();self.trace.zero_();self.solver_info.zero_()
    @torch.no_grad()
    def step(self,length):
        cos,sin=self.local_geometry[length]
        z=self.x[:length]
        prediction,cache=forward(z,self.layers,cos,sin)
        rhs=self.target[:length]-prediction;rhs[0]=0
        delta,projected,info=direction(lambda value:diagonal_jvp(value,self.layers,cache,cos,sin),rhs,self.k)
        norm=delta.norm(dim=-1)
        factor=(self.radius/norm.clamp_min(1e-20)).clamp(max=1)
        error=(prediction[1:]-self.target[1:length]).square().mean()
        cosine=(1-F.cosine_similarity(prediction[1:],self.target[1:length],dim=-1)).mean()
        relative=(projected[1:]/rhs[1:].norm(dim=-1).clamp_min(1e-20)).mean()
        record=torch.stack([error,cosine,norm[1:].mean(),(factor[1:]<1).float().mean(),relative])
        self.trace.index_copy_(0,self.counter,record[None]);self.counter.add_(1)
        self.solver_info[:length].copy_(torch.maximum(self.solver_info[:length],info.abs()))
        z.add_(self.damping*factor[:,None]*delta);z[0].copy_(self.weight[128000])
    @torch.no_grad()
    def ensure_local(self,length):
        if length in self.local_graphs:
            self.local_graphs.move_to_end(length);return 0.
        torch.cuda.synchronize();start=time.perf_counter()
        if len(self.local_graphs)>=2:
            old,graph=self.local_graphs.popitem(last=False);del graph
            self.local_geometry.pop(old);gc.collect();torch.cuda.empty_cache()
        position=torch.arange(length,device="cuda")[None]
        cos,sin=self.prefix.rotary_emb(self.x[None,:length],position)
        self.local_geometry[length]=(cos[0],sin[0])
        self.x[:length].copy_(self.weight[128000].expand(length,-1))
        target,_=forward(self.x[:length],self.layers,cos[0],sin[0]);self.target[:length].copy_(target)
        self.counter.zero_();self.trace.zero_();self.solver_info.zero_()
        stream=self.local_stream;stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.step(length)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.counter.zero_()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):self.step(length)
        torch.cuda.synchronize()
        self.local_graphs[length]=graph
        elapsed=time.perf_counter()-start
        self.local_capture_events.append({"length":length,"seconds":elapsed,"reserved_bytes":torch.cuda.memory_reserved()})
        return elapsed
    def decode(self,h,replay=True):
        torch.cuda.synchronize();start=time.perf_counter();length=len(h)
        local_capture=self.ensure_local(length)
        warm,warm_stats=self.soft.decode(h)
        with torch.no_grad():
            probability=F.softmax(self.soft.logits[:length-1],dim=-1)
            mean=BF16Mixture.apply(probability,self.soft.weight)
            self.reset_local(mean,h)
            out={"warm_"+key:value for key,value in warm.items()}
            out["initial_embedding"]=mean.cpu()
        torch.cuda.synchronize();local_start=time.perf_counter();checkpoint_times={}
        for iteration in range(16):
            if replay:self.local_graphs[length].replay()
            else:self.step(length)
            if iteration+1 in CHECKPOINTS:
                with torch.no_grad():
                    scores=self.score_tables(self.x[1:length])
                    for name,value in scores.items():
                        out[f"step{iteration+1}_{name}"]=torch.cat([torch.full((1,),128000,device="cuda",dtype=torch.long),value.argmax(-1)]).cpu()
                torch.cuda.synchronize();checkpoint_times[str(iteration+1)]=time.perf_counter()-start
        with torch.no_grad():
            cos,sin=self.local_geometry[length]
            final,_=forward(self.x[:length],self.layers,cos,sin)
            out["embedding_final"]=self.x[:length].cpu();out["local_trace"]=self.trace.cpu()
            out["final_position_mse"]=(final[1:]-self.target[1:length]).square().mean(-1).cpu()
            out["final_position_cosine_error"]=(1-F.cosine_similarity(final[1:],self.target[1:length],dim=-1)).cpu()
            out["solver_info"]=self.solver_info[:length].cpu()
            finite=all(bool(torch.isfinite(value).all()) for value in out.values())
            ok_info=bool((out["solver_info"]==0).all())
        torch.cuda.synchronize();end=time.perf_counter()
        return out,{"total_seconds":end-start,"warm":warm_stats,"local_and_readout_seconds":end-local_start,
          "local_capture_seconds":local_capture,"checkpoint_total_seconds_all_four_readouts":checkpoint_times,
          "nonlinear_full_sequence_forwards":16+1,"own_position_jvp_calls":16*self.k,
          "warm_prefix_forwards":warm_stats["whole_sequence_prefix_forwards"],"warm_prefix_backwards":warm_stats["whole_sequence_prefix_backwards"],
          "readout_vocabulary_products":2*len(CHECKPOINTS),"additional_mixture_products":1,
          "vocabulary_entries_per_sweep":len(self.weight),"shortlist_size":None,"separate_candidate_verification_calls":0,"model_parameter_updates":0,
          "numeric_valid":finite and ok_info,"direction_radius":float(self.radius),"ridge":1e-6,"krylov_steps":self.k,"damping":self.damping,
          "local_numerics":"FP32_TF32_DISABLED; analytic own-position Jacobian; solve_ex; block-Jacobi nonlinear updates",
          "trace_columns":["pre_update_mse","pre_update_cosine_error","unclipped_direction_norm","clipped_position_fraction","projected_linear_residual_ratio"],
          "timing_scope":"warm plus16local updates, three sets of all four readouts, final continuous forward and output transfer; preparation/capture recorded separately"}

"""Sequential continuous inversion using emitted tokens as immutable history."""
import time,torch
from torch.nn import functional as F
from mean_readout import MeanReadout
from fast_soft import BF16Mixture
from linearized_prefix import public_parameters
from single_position import forward,jvp,vjp,commit
from normalized import equation
from cheap_update import direction
CONFIGS=[(f"warm{warm}_{rule}_updates{updates}",warm,rule,updates)
         for warm in [0,64] for rule in ["polyak0.25","cg1"] for updates in [1,4]]
AUX_KEYS={"initial_embedding","embedding_final","first_prediction","local_trace","cache_lengths","committed_tokens"}

class SequentialInverse(MeanReadout):
    def __init__(self,prefix,warm,rule,updates):
        super().__init__(prefix,warm)
        if rule not in ["polyak0.25","cg1"] or updates not in [1,4]:raise ValueError("unregistered rule")
        self.prefix=prefix;self.rule=rule;self.updates=updates
        self.layers=public_parameters(prefix);self.radius=self.raw_norm.sqrt().median()
    def decode(self,h):
        torch.cuda.synchronize();start=time.perf_counter();phase={}
        def timed(name,fn):
            t=time.perf_counter();value=fn();torch.cuda.synchronize()
            phase[name]=phase.get(name,0.)+time.perf_counter()-t
            return value
        warm,ws=timed("warm",lambda:self.soft.decode(h))
        output,details=self.current(h,warm,timed)
        torch.cuda.synchronize();end=time.perf_counter()
        length=len(h);count=(length-1)*self.updates
        total=end-start
        stats={"total_seconds":total,"phase_seconds":phase,"warm":ws,
          "unattributed_host_seconds":total-sum(v for k,v in phase.items() if k not in ["jvp","vjp"]),
          "current_forwards":count,"current_vjp_calls":count,"current_jvp_calls":count if self.rule=="cg1" else 0,
          "committed_token_forwards":length-1,"final_token_commit_skipped":True,
          "warm_prefix_forwards":ws["whole_sequence_prefix_forwards"],"warm_prefix_backwards":ws["whole_sequence_prefix_backwards"],
          "additional_mixture_products":1,"readout_vocabulary_products":2*(length-1),
          "vocabulary_entries_per_sweep":len(self.weight),"shortlist_size":None,
          "separate_candidate_verification_calls":0,"model_parameter_updates":0,
          "current_updates":self.updates,"rule":self.rule,"direction_radius":float(self.radius),
          "numeric_valid":all(bool(torch.isfinite(v).all()) for v in output.values()),
          "current_numerics":"uncaptured FP32; variable native past length; committed emitted IDs only; TF32 disabled",
          "trace_columns":["pre_update_mse","pre_update_cosine_error","unclipped_direction_norm","clip_factor"],
          "timing_scope":"end-to-end warm, input/mixture, current forwards, directions, all four readout formula calculations, cache commits and output transfer; synchronization charged to each phase; JVP/VJP sub-times overlap direction total",
          **details}
        return output,stats
    @torch.no_grad()
    def current(self,h,warm,timed):
        length=len(h)
        def prepare():
            target=h.to("cuda").float()
            probability=F.softmax(self.soft.logits[:length-1],dim=-1)
            mean=BF16Mixture.apply(probability,self.soft.weight)
            position=torch.arange(length,device="cuda")[None]
            cos,sin=self.prefix.rotary_emb(mean[None],position)
            return target,mean,cos[0],sin[0]
        target,mean,cos,sin=timed("input_mixture_geometry",prepare)
        ids=torch.full((length,),128000,device="cuda",dtype=torch.long)
        final=torch.empty((length-1,self.weight.shape[1]),device="cuda")
        first=torch.empty_like(final)
        trace=torch.empty((length-1,self.updates,4),device="cuda")
        lengths=[];committed=[]
        past=[(torch.empty((p["kv_heads"],0,p["head_dim"]),device="cuda"),
               torch.empty((p["kv_heads"],0,p["head_dim"]),device="cuda")) for p in self.layers]
        def commit_position(pos,history):
            # The activation is deliberately discarded: an emitted ID cannot be revised.
            _,_,current=forward(self.weight[ids[pos:pos+1]],self.layers,history,cos[pos:pos+1],sin[pos:pos+1])
            return commit(history,current)
        past=timed("commit",lambda:commit_position(0,past));committed.append(ids[0:1].clone())
        for pos in range(1,length):
            history_lengths=[(a.shape[1],b.shape[1]) for a,b in past]
            if any(v!=(pos,pos) for v in history_lengths):raise RuntimeError("invalid causal cache length")
            lengths.append(pos);z=mean[pos-1:pos].clone();co,si=cos[pos:pos+1],sin[pos:pos+1]
            for step in range(self.updates):
                prediction,caches,_=timed("current_forward",lambda:forward(z,self.layers,past,co,si))
                if step==0:first[pos-1].copy_(prediction[0])
                def update():
                    raw_mv=lambda value:timed("jvp",lambda:jvp(value,self.layers,caches,co,si))
                    raw_rmv=lambda value:timed("vjp",lambda:vjp(value,self.layers,caches,co,si))
                    rhs,mv,rmv=equation(prediction,target[pos:pos+1],raw_mv,raw_rmv)
                    # This one row is an unknown position. It must not be zeroed as BOS.
                    delta,work=direction(self.rule,rhs,mv,rmv)
                    norm=delta.norm(dim=-1)
                    factor=(self.radius/norm.clamp_min(1e-20)).clamp(max=1)
                    record=torch.stack([(prediction-target[pos:pos+1]).square().mean(),
                        1-F.cosine_similarity(prediction,target[pos:pos+1],dim=-1)[0],norm[0],factor[0]])
                    trace[pos-1,step].copy_(record)
                    return z+factor[:,None]*delta
                z=timed("direction",update)
            def readout():
                # Preserve the existing formula and count its two vocabulary products.
                scores=self.score_tables(z)
                return scores["white_cosine"].argmax(-1)
            ids[pos:pos+1].copy_(timed("readout",readout))
            final[pos-1].copy_(z[0])
            if pos<length-1:
                past=timed("commit",lambda:commit_position(pos,past))
                committed.append(ids[pos:pos+1].clone())
        def transfer():
            result={"warm_"+k:v for k,v in warm.items()}
            result.update(tokens=ids.cpu(),initial_embedding=mean.cpu(),embedding_final=final.cpu(),
              first_prediction=first.cpu(),local_trace=trace.cpu(),
              cache_lengths=torch.tensor(lengths,dtype=torch.int64),
              committed_tokens=torch.cat(committed).cpu())
            return result
        return timed("output_transfer",transfer),{"causal_cache_checks":length-1}

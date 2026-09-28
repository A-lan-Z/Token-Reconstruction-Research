"""Full causal, probe-scaled normalized inverse with unrestricted vocabulary readout."""
import time,torch
from torch.nn import functional as F
from inverse import LocalInverse,CHECKPOINTS,READOUTS
from linearized_prefix import forward
from causal import operators,free_positions
from normalized import equation
from global_cgls import least_squares
from precondition import make_scale,scaled_operators
CONFIGS=[(f"warm{warm}_cg{k}_damp{damping}",warm,k,damping)
         for warm in [0,32,64] for k in [8,16,32] for damping in [.5,1.]]
class CoupledInverse(LocalInverse):
    def __init__(self,prefix,warm,k,damping):
        super().__init__(prefix,warm,k,damping)
        self.fixed_probes={};self.probe_preparation_events=[]
    @torch.no_grad()
    def ensure_local(self,length):
        if length not in self.fixed_probes:
            torch.cuda.synchronize();start=time.perf_counter()
            generator=torch.Generator().manual_seed(200063+length)
            self.fixed_probes[length]=tuple((torch.randint(0,2,(length,2048),generator=generator)*2-1).to(device="cuda",dtype=torch.float32) for _ in range(4))
            torch.cuda.synchronize();self.probe_preparation_events.append({"length":length,"seconds":time.perf_counter()-start,"seed":200063+length,"probes":4})
        return super().ensure_local(length)
    @torch.no_grad()
    def step(self,length):
        cos,sin=self.local_geometry[length];z=self.x[:length]
        prediction,cache=forward(z,self.layers,cos,sin)
        raw_mv,raw_rmv=operators(self.layers,cache,cos,sin)
        rhs,mv,rmv=equation(prediction,self.target[:length],raw_mv,raw_rmv);rhs=free_positions(rhs)
        scale,_=make_scale("coordinate_probe4",z,rmv,self.fixed_probes[length])
        j,jt=scaled_operators(mv,rmv,scale)
        value,stats=least_squares(j,jt,rhs,self.k,.0001);delta=scale*value
        norm=delta.norm(dim=-1);factor=(self.radius/norm.clamp_min(1e-20)).clamp(max=1)
        mse=(prediction[1:]-self.target[1:length]).square().mean()
        cosine=(1-F.cosine_similarity(prediction[1:],self.target[1:length],dim=-1)).mean()
        ratio=stats["global_recurrence_residual_norm"]/rhs.norm().clamp_min(1e-20)
        record=torch.stack([mse,cosine,norm[1:].mean(),(factor[1:]<1).float().mean(),ratio])
        self.trace.index_copy_(0,self.counter,record[None]);self.counter.add_(1)
        z.add_(self.damping*factor[:,None]*delta);z[0].copy_(self.weight[128000])
    def decode(self,h,replay=True):
        before=len(self.probe_preparation_events)
        output,stats=super().decode(h,replay=replay)
        for key in ["ridge","krylov_steps","own_position_jvp_calls"]:stats.pop(key)
        stats.update(objective="cosine_via_normalized_squared_residual",linear_iterations=self.k,
          linear_ridge_multiplier=.0001,linear_ridge_rule="multiplier * global ||(JnD)^T rhs||^2 / ||rhs||^2; scaled-coordinate regularizer",
          full_causal_jvp_calls=16*self.k,full_causal_vjp_calls=16*(self.k+1+4),
          scaling="coordinate_probe4",scale_probe_calls=16*4,scale_probe_seed=200063+len(h),
          probe_preparation_seconds=sum(v["seconds"] for v in self.probe_preparation_events[before:]),
          local_numerics="FULL_CAUSAL_NORMALIZED_CGLS_FP32_TF32_DISABLED; global coefficients; BOS fixed; coordinate-probe right scaling",
          trace_columns=["pre_update_mse","pre_update_cosine_error","unclipped_direction_norm","clipped_position_fraction","global_recurrence_linear_residual_ratio"])
        return output,stats

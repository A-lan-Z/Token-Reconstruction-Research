"""Normalized local least-squares reconstruction with full-vocabulary readout."""
import torch
from torch.nn import functional as F
from inverse import LocalInverse,CHECKPOINTS,READOUTS
from linearized_prefix import forward,diagonal_jvp
from adjoint import diagonal_vjp
from cgls import least_squares
from normalized import equation
CONFIGS=[(f"warm{warm}_cg{k}_ridge{ridge}",warm,k,ridge)
         for warm in [0,32,64] for k in [4,8] for ridge in [.0001,.01]]
class CosineInverse(LocalInverse):
    def __init__(self,prefix,warm,k,ridge):
        super().__init__(prefix,warm,k,.5);self.ridge_multiplier=ridge
    @torch.no_grad()
    def step(self,length):
        cos,sin=self.local_geometry[length];z=self.x[:length]
        prediction,cache=forward(z,self.layers,cos,sin)
        rhs,mv,rmv=equation(prediction,self.target[:length],
          lambda v:diagonal_jvp(v,self.layers,cache,cos,sin),
          lambda v:diagonal_vjp(v,self.layers,cache,cos,sin))
        rhs[0]=0
        delta,stats=least_squares(mv,rmv,rhs,self.k,self.ridge_multiplier)
        norm=delta.norm(dim=-1);factor=(self.radius/norm.clamp_min(1e-20)).clamp(max=1)
        mse=(prediction[1:]-self.target[1:length]).square().mean()
        cosine=(1-F.cosine_similarity(prediction[1:],self.target[1:length],dim=-1)).mean()
        ratio=(stats["recurrence_residual_norm"][1:]/rhs[1:].norm(dim=-1).clamp_min(1e-20)).mean()
        record=torch.stack([mse,cosine,norm[1:].mean(),(factor[1:]<1).float().mean(),ratio])
        self.trace.index_copy_(0,self.counter,record[None]);self.counter.add_(1)
        z.add_(self.damping*factor[:,None]*delta);z[0].copy_(self.weight[128000])
    def decode(self,h,replay=True):
        output,stats=super().decode(h,replay=replay)
        stats.pop("ridge");stats.pop("krylov_steps")
        stats.update(objective="cosine_via_normalized_squared_residual",linear_iterations=self.k,linear_ridge_multiplier=self.ridge_multiplier,
          linear_ridge_rule="multiplier * ||Jn^T rhs||^2 / ||rhs||^2, fixed per normalized linear solve",
          own_position_jvp_calls=16*self.k,own_position_vjp_calls=16*(self.k+1),
          local_numerics="NORMALIZED_CGLS_FP32_TF32_DISABLED; analytic own-position normalized J/JT; block-Jacobi continuous updates",
          trace_columns=["pre_update_mse","pre_update_cosine_error","unclipped_direction_norm","clipped_position_fraction","recurrence_linear_residual_ratio"])
        return output,stats

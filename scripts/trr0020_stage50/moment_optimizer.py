"""Reconstruction with a full-vocabulary moment-bound exponential step."""
import torch
from reused_optimizer import ReusedOptimizer
from moment_step import update,ROOT_ITERATIONS
class MomentOptimizer(ReusedOptimizer):
    def __init__(self,prefix,mode):
        if mode not in ["control","bennett","two_point"]:raise ValueError("unknown rule")
        self.moment_mode=mode
        self.validate_moment_kl=False;self.moment_kl_checks=[]
        super().__init__(prefix,True)
    def train_step(self,length):
        if self.stage=="warm" or self.moment_mode=="control":return super().train_step(length)
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        with torch.no_grad():
            new,info=update(self.logits[:length-1],self.gradient[:length-1],self.position_error[:length-1],
              self.moment_mode,probability=self.reused_probability)
            if self.validate_moment_kl and int(self.counter.item()) in [1,9,33,65,128]:
                lp=self.logits[:length-1].detach().cpu().double().log_softmax(-1)
                lq=new.detach().cpu().double().log_softmax(-1)
                actual=(lp.exp()*(lp-lq)).sum(-1);budget_ref=info["requested_budget"].cpu().double()
                passed=bool(torch.isfinite(actual).all() and (actual>=-2e-5).all() and (actual<=budget_ref+2e-5).all())
                self.moment_kl_checks.append({"step":int(self.counter.item()),"direct_kl":actual.tolist(),
                  "budget":budget_ref.tolist(),"bound":info["bound_kl"].cpu().tolist(),"passed":passed})
                if not passed:raise RuntimeError("public trajectory KL feasibility failed")
            self.logits[:length-1].copy_(new)
            bound=info["bound_kl"].float();budget=info["requested_budget"]
            used=bound/budget.clamp_min(torch.finfo(budget.dtype).tiny)
            record=torch.stack([budget.mean(),bound.mean(),used.mean(),info["normalized_step"].amax(),info["active"].float().mean()])
            self.mirror_update_trace.index_copy_(0,self.counter-1,record[None])
    def decode(self,h,steps=64,replay=True):
        out,stats=super().decode(h,steps,replay)
        stats["moment_rule"]=self.moment_mode
        stats["moment_root_iterations"]=0 if self.moment_mode=="control" else ROOT_ITERATIONS
        if self.moment_mode!="control":
            stats["scalar_iterations"]=0
            stats["update_implementation"]="full_vocabulary_moment_bound_fp64_scalar_root"
            stats["update_trace_columns"]=["mean_requested_budget","mean_upper_kl_bound","mean_bound_budget_fraction","max_normalized_step","active_fraction"]
            stats["full_vocabulary_kl_root_evaluations_per_update"]=0
            stats["bound_semantics"]="mathematical moment upper bound; finite precision qualified against independent direct KL on fixed public states"
        else:stats["full_vocabulary_kl_root_evaluations_per_update"]=5
        return out,stats

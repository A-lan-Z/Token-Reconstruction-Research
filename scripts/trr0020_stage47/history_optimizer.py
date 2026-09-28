"""Input-specific bounded history mixing; no candidate proposal or learned predictor."""
import time,torch
from reused_optimizer import ReusedOptimizer
from reused_step import update
from aa_step import mix
CONFIGS={"control":(None,0.),"aa_logit_clip1":("logit",1.),"aa_probability_clip1":("probability",1.),"aa_probability_clip2":("probability",2.)}
class HistoryOptimizer(ReusedOptimizer):
    def __init__(self,prefix,name):
        self.name=name;self.metric,self.clip=CONFIGS[name]
        super().__init__(prefix,True)
        self.old_residual=torch.zeros_like(self.logits);self.old_proposal=torch.zeros_like(self.logits)
        self.history_valid=torch.zeros((),dtype=torch.bool,device="cuda")
        self.aa_trace=torch.zeros(1025,3,device="cuda")
    def reset_mirror_statistics(self):
        super().reset_mirror_statistics()
        if hasattr(self,"history_valid"):
            self.history_valid.zero_();self.old_residual.zero_();self.old_proposal.zero_();self.aa_trace.zero_()
    def train_step(self,length):
        if self.stage=="warm" or self.metric is None:return super().train_step(length)
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        with torch.no_grad():
            current=self.logits[:length-1]
            proposal,info=update(current,self.gradient[:length-1],self.position_error[:length-1],2.,4,probability=self.reused_probability)
            new,residual,aa=mix(current,proposal,self.reused_probability,self.old_residual[:length-1],
              self.old_proposal[:length-1],self.history_valid,self.metric,self.clip)
            self.old_residual[:length-1].copy_(residual);self.old_proposal[:length-1].copy_(proposal);self.history_valid.fill_(True)
            current.copy_(new)
            record=torch.stack([info["requested_budget"].mean(),info["formula_kl"].mean(),info["budget_used"].mean(),
              info["normalized_step"].amax(),info["active"].float().mean()])
            self.mirror_update_trace.index_copy_(0,self.counter-1,record[None])
            ar=torch.stack([aa["coefficient"],aa["residual_squared_norm"],aa["denominator"]])
            self.aa_trace.index_copy_(0,self.counter-1,ar[None])
    def decode(self,h,steps=64,replay=True):
        out,stats=super().decode(h,steps,replay)
        start=time.perf_counter()
        if self.metric is not None:out["aa_trace"]=self.aa_trace[:steps].cpu()
        extra=time.perf_counter()-start
        stats["history_output_transfer_seconds"]=extra
        stats["total_seconds"]+=extra
        stats.update(history_rule=self.name,history_ridge=1e-4,history_metric=self.metric,history_clip=self.clip,
          history_scope="one global scalar from current/previous full-vocabulary residuals; logits mix; no fitted predictor")
        if self.metric is not None:stats["budget_rule"]="base update uses original KL budget; subsequent bounded history mixing has no claimed KL bound"
        return out,stats

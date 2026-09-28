"""Cold full-vocabulary optimization with observed-error KL budgets."""
import time,torch
from torch.nn import functional as F
from mirror_optimizer import MirrorOptimizer as Base,AUX_KEYS,gradient_reference
from fast_soft import BF16Mixture
from discrete_soft import DiscreteSoftVocabulary
from budget_step import update
CONFIGS=[(f"cold_factor{factor}_iter{iterations}",factor,iterations) for factor in [.5,1.,2.] for iterations in [4,8]]
CONFIGS.append(("cold_factor2.0_iter16",2.,16))
CHECKPOINTS=(0,1,2,4,8,16,32,64)
class BudgetOptimizer(Base):
    def __init__(self,prefix,factor,iterations):
        super().__init__(prefix,0,.01)
        self.factor=factor;self.scalar_iterations=iterations
        self.validate_kl=False;self.kl_references=[]
    def train_step(self,length):
        if self.stage=="warm":return super().train_step(length)
        self.gradient.zero_();loss=self.evaluate(length);loss.backward()
        with torch.no_grad():
            new,info=update(self.logits[:length-1],self.gradient[:length-1],
                self.position_error[:length-1],self.factor,self.scalar_iterations)
            if self.validate_kl and int(self.counter.item()) in [1,9,33,64]:
                a=self.logits[:length-1].detach().cpu().double().log_softmax(-1)
                b=new.cpu().double().log_softmax(-1)
                actual=(a.exp()*(a-b)).sum(-1)
                budget=info["requested_budget"].cpu().double()
                self.kl_references.append({"step":int(self.counter.item()),
                  "direct_kl":actual.tolist(),"budget":budget.tolist(),
                  "passed":bool(torch.isfinite(actual).all() and (actual<=budget+2e-5).all() and (actual>=-2e-5).all())})
            self.logits[:length-1].copy_(new)
            record=torch.stack([info["requested_budget"].mean(),info["formula_kl"].mean(),
                info["budget_used"].mean(),info["normalized_step"].amax(),info["active"].float().mean()])
            self.mirror_update_trace.index_copy_(0,self.counter-1,record[None])
    def decode(self,h,replay=True):
        self.stage="warm";torch.cuda.synchronize();start=time.perf_counter();length=len(h)
        warm_capture=self.ensure(length);mirror_capture=self.ensure_mirror(length)
        warm,ws=DiscreteSoftVocabulary.decode(self,h);torch.cuda.synchronize();warm_end=time.perf_counter()
        with torch.no_grad():
            p=F.softmax(self.logits[:length-1],dim=-1)
            mean=BF16Mixture.apply(p,self.weight)
            out={"warm_"+k:v for k,v in warm.items()};out["initial_embedding"]=mean.cpu()
            self.reset_mirror_statistics()
        self.stage="mirror";torch.cuda.synchronize();mirror_start=time.perf_counter();checkpoints={}
        training,evaluating=self.mirror_graphs[length]
        for step in range(self.mirror_steps):
            if replay:training.replay()
            else:self.train_step(length)
            if step in CHECKPOINTS:
                out["step"+str(step)]=self.tokens[:length].cpu()
                torch.cuda.synchronize();checkpoints[str(step)]=time.perf_counter()-start
        if replay:evaluating.replay()
        else:
            with torch.no_grad():self.evaluate(length)
        with torch.no_grad():
            out["step64"]=self.tokens[:length].cpu()
            out["best_objective"]=self.best_tokens[:length].cpu()
            out["best_position_error"]=self.position_best_tokens[:length].cpu()
            out["mirror_update_trace"]=self.mirror_update_trace[:64].cpu()
            out["final_confidence"]=self.confidence[:length-1].cpu()
            out["final_position_error"]=self.position_error[:length-1].cpu()
            traces={"loss":self.loss_trace[:65].cpu().tolist(),"confidence":self.confidence_trace[:65].cpu().tolist(),
              "gini":self.purity_trace[:65].cpu().tolist()}
            valid=bool(torch.isfinite(self.logits[:length-1]).all()) and all(bool(torch.isfinite(v).all()) for v in out.values())
        torch.cuda.synchronize();end=time.perf_counter();checkpoints["64"]=end-start;self.stage="warm"
        return out,{"total_seconds":end-start,"warm":ws,"warm_capture_seconds":warm_capture,"mirror_capture_seconds":mirror_capture,
          "warm_stage_seconds_including_capture":warm_end-start,"mirror_seconds":end-mirror_start,
          "mixture_transfer_and_reset_seconds":mirror_start-warm_end,"checkpoint_total_seconds":checkpoints,
          "mirror_traces":traces,"mirror_steps":64,"factor":self.factor,"scalar_iterations":self.scalar_iterations,
          "numeric_valid":valid,"whole_sequence_prefix_forwards":self.steps+1+65,"whole_sequence_prefix_backwards":self.steps+64,
          "additional_mixture_products":1,"vocabulary_entries_per_sweep":self.vocab,"shortlist_size":None,
          "separate_candidate_verification_calls":0,"model_parameter_updates":0,
          "update_trace_columns":["mean_requested_budget","mean_formula_kl","mean_budget_used","max_normalized_step","active_fraction"],
          "numerics":"existing BF16 surrogate gradient; full-vocabulary exponential path with observed-error KL budget and fixed safeguarded iterations; TF32 disabled",
          "budget_rule":"clamp(factor * current observed cosine error,0,1)",
          "timing_scope":"warm plus64mirror steps,65mirror evaluations, diagnostic outputs, synchronization and transfers; first-input captures included and separately recorded"}

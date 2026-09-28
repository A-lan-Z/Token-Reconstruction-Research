"""Armijo backtracking along a full-vocabulary KL-calibrated exponential path."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage32"))
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage33"))
from budget_step import path_update
from probability_gradient import gradient
ARMIJO=1e-4
SHRINK=.25
GROW=2.
MAX_TRIALS=4
FACTOR=2.
CAP=1.
ROOT_ITERATIONS=4
STEP_COLUMNS=["loss_before","loss_after","confidence_after","budget_start","accepted_budget","accepted_formula_kl","trials","accepted","next_budget","accepted_linear_change"]
TRIAL_COLUMNS=["step","trial","loss_before","loss_trial","linear_change","budget","formula_kl","accepted","armijo_rhs"]
def error(output,target):
    unit=output/output.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    target_unit=target/target.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return 1-(unit*target_unit).sum(-1)
@torch.no_grad()
def solve(z0,target,E,evaluate,input_vjp,steps):
    if z0.ndim!=2 or len(z0)!=1 or z0.shape[1]!=len(E) or steps<1:raise ValueError("one full-vocabulary current token required")
    z=z0.clone();p=z.softmax(-1);mean=p@E;out,cache=evaluate(mean);loss=error(out,target)
    radius=(FACTOR*loss.clamp_min(0)).clamp(max=CAP)
    records=[];trials=[];states=[];forward_count=1;gradient_count=0;accepted_count=0
    scalar=lambda value:z.new_tensor(value)
    for step in range(steps):
        G,gi=gradient(out,target,E,lambda vector:input_vjp(vector,cache))
        gradient_count+=1;oldloss=gi["loss"];start_radius=radius.clone();budget=radius.clone()
        accepted=False;used_budget=torch.zeros_like(radius);used_kl=torch.zeros_like(radius);used_prediction=torch.zeros_like(radius)
        trial_count=0
        for attempt in range(MAX_TRIALS):
            new,info=path_update(z,G,budget,ROOT_ITERATIONS)
            q=new.softmax(-1);newmean=q@E;newout,newcache=evaluate(newmean);newloss=error(newout,target)
            forward_count+=1;trial_count+=1
            linear=(G*(q-p)).sum(-1);rhs=oldloss+ARMIJO*linear
            take=bool(((linear<=0)&(newloss<=rhs)&torch.isfinite(newloss)&torch.isfinite(linear)).all())
            states.append((z.clone(),new.clone(),budget.clone()))
            trials.append(torch.stack([scalar(step),scalar(attempt),oldloss[0],newloss[0],linear[0],budget[0],info["formula_kl"][0],scalar(take),rhs[0]]))
            if take:
                z,p,mean,out,cache,loss=new,q,newmean,newout,newcache,newloss
                used_budget=budget;used_kl=info["formula_kl"];used_prediction=linear
                radius=torch.minimum(GROW*budget,(FACTOR*newloss.clamp_min(0)).clamp(max=CAP))
                accepted=True;accepted_count+=1
                break
            budget=budget*SHRINK
        if not accepted:
            loss=oldloss;radius=budget
        records.append(torch.stack([oldloss[0],loss[0],p.amax(),start_radius[0],used_budget[0],used_kl[0],scalar(trial_count),scalar(accepted),radius[0],used_prediction[0]]))
    data={"logits":z,"mixture":mean,"prefix_output":out,"observed_error":loss,
      "confidence":p.amax(-1),"token":z.argmax(-1),"step_trace":torch.stack(records),"trial_trace":torch.stack(trials)}
    counts={"prefix_forwards":forward_count,"prefix_vjps":gradient_count,"vocabulary_mixture_products":forward_count,
      "vocabulary_gradient_products":gradient_count,"probability_updates":len(trials),"accepted_steps":accepted_count,
      "rejected_trials":len(trials)-accepted_count,"vocabulary":len(E),"shortlist":None,"model_parameter_updates":0}
    return data,states,counts

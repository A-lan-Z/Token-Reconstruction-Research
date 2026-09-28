"""Whole-sequence optimization with one evolving estimate and no shortlist."""
import time
import torch
import torch.nn.functional as F
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric

class JointProjection:
    def __init__(self,prefix):
        self.prefix=prefix
        self.metric=PrefixWeightMetric(prefix);self.metric.build()
        self.weight=prefix.embed_tokens.weight.detach().float()
        self.normalized=F.normalize(self.weight,dim=-1)
        self.norms=self.weight.square().sum(-1)
        self.bos=self.weight[128000].view(1,1,-1)
        self.scale=self.metric.transform.square().sum().sqrt()/self.weight.shape[1]**.5
        self.transform=self.metric.transform/self.scale
    @torch.no_grad()
    def project(self,z):
        ids=(F.normalize(z.reshape(-1,z.shape[-1]),dim=-1)@self.normalized.T).argmax(-1)
        ids[0]=128000
        return ids
    def forward(self,z,pe,mask):
        h=z.to(torch.bfloat16)
        for layer in self.prefix.layers:
            h=self.prefix._hidden(layer(h,position_embeddings=pe,attention_mask=mask,use_cache=False))
        return h.float()
    def decode(self,observation,method):
        self.metric._check()
        torch.cuda.synchronize();start=time.perf_counter()
        h=observation.to("cuda").float().unsqueeze(0)
        ids=self.metric.propose(h[0],1).flatten();ids[0]=128000
        z=self.weight[ids[1:]].unsqueeze(0).clone().requires_grad_()
        pos=torch.arange(h.shape[1],device="cuda").view(1,-1)
        pe=self.prefix.rotary_emb(h.to(torch.bfloat16),pos)
        mask=self.prefix._causal_mask(h.to(torch.bfloat16),start_pos=0,total_tokens=h.shape[1])
        kind,losskind,steps,snap=method.split("_");steps=int(steps);snap=int(snap)
        if kind=="adam":
            opt=torch.optim.Adam([z],lr=.003)
        else:
            opt=torch.optim.LBFGS([z],lr=1,max_iter=steps,max_eval=2*steps,
                tolerance_grad=1e-8,tolerance_change=1e-10,history_size=8,line_search_fn="strong_wolfe")
        trace=[];best=[float("inf"),z.detach().clone()];projections=1
        def closure():
            opt.zero_grad(set_to_none=True)
            predicted=self.forward(torch.cat([self.bos,z],dim=1),pe,mask)
            residual=predicted[:,1:]-h[:,1:]
            if losskind=="white":residual=residual@self.transform
            loss=residual.square().sum()/z.shape[1]
            if not torch.isfinite(loss):raise RuntimeError("nonfinite optimization loss")
            loss.backward()
            if not torch.isfinite(z.grad).all():raise RuntimeError("nonfinite optimization gradient")
            value=float(loss.detach());trace.append(value)
            if value<best[0]:best[:]=[value,z.detach().clone()]
            return loss
        if kind=="adam":
            for i in range(steps):
                closure();opt.step()
                if snap and (i+1)%snap==0:
                    with torch.no_grad():
                        ids=self.project(torch.cat([self.bos,z],dim=1))
                        z.copy_(self.weight[ids[1:]].unsqueeze(0));opt.state.clear();projections+=1
        else:opt.step(closure)
        torch.cuda.synchronize();solved=time.perf_counter()
        tokens=self.project(torch.cat([self.bos,best[1]],dim=1)).cpu()
        torch.cuda.synchronize();end=time.perf_counter()
        return {"tokens":tokens},{"total":end-start,"solve_and_initialization":solved-start,
               "final_projection_and_output":end-solved,"loss":trace,
               "continuous_sequence_forward_backward_equivalents":len(trace),
               "discrete_candidate_simulations":0,"full_vocabulary_projections":projections+1}


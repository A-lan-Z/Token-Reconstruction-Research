"""Discrete whole-sequence straight-through refinement, no shortlist."""
import time
import torch
import torch.nn.functional as F
from joint_projection import JointProjection
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv

class DiscreteParallel(JointProjection):
    def diagonal_forward(self,z,pe,mask):
        """Same causal structure; detach other positions' K/V in gradients.
        Eager attention arithmetic is an explicit new numerical method."""
        h=z.to(torch.bfloat16)
        for ly in self.prefix.layers:
            inp=ly.input_layernorm(h);att=ly.self_attn
            sh=(*inp.shape[:-1],-1,att.head_dim)
            q=att.q_proj(inp).view(sh).transpose(1,2)
            k=att.k_proj(inp).view(sh).transpose(1,2)
            v=att.v_proj(inp).view(sh).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,*pe)
            k=repeat_kv(k,att.num_key_value_groups);v=repeat_kv(v,att.num_key_value_groups)
            # Keep the query path, plus only same-position key/value derivatives.
            score=(q@k.detach().transpose(-1,-2))*att.scaling
            diag=(q*(k-k.detach())).sum(-1)*att.scaling
            score=score+torch.diag_embed(diag)
            if mask is not None:score=score+mask
            a=F.softmax(score.float(),dim=-1).to(q.dtype)
            out=a@v.detach()+a.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*(v-v.detach())
            out=out.transpose(1,2).reshape(*inp.shape).contiguous()
            h=h+att.o_proj(out)
            h=h+ly.mlp(ly.post_attention_layernorm(h))
        return h.float()

    def decode(self,observation,method):
        self.metric._check();torch.cuda.synchronize();start=time.perf_counter()
        h=observation.to("cuda").float().unsqueeze(0)
        lossname,gradname,steps,lr=method.split("_");steps=int(steps);lr=float(lr)
        ids=self.metric.propose(h[0],1).flatten();ids[0]=128000
        z=self.weight[ids[1:]].unsqueeze(0).clone().requires_grad_()
        pos=torch.arange(h.shape[1],device="cuda").view(1,-1)
        pe=self.prefix.rotary_emb(h.to(torch.bfloat16),pos)
        mask=self.prefix._causal_mask(h.to(torch.bfloat16),start_pos=0,total_tokens=h.shape[1])
        opt=torch.optim.Adam([z],lr=lr)
        bestloss=float("inf");bestids=ids.clone();trace=[];changes=[]
        forward=self.diagonal_forward if gradname=="diagonal" else self.forward
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            zfull=torch.cat([self.bos,z],dim=1)
            ids=self.project(zfull)
            # Forward exactly on vocabulary rows; gradient on proxy vector.
            discrete=self.weight[ids].unsqueeze(0)
            value=discrete+(zfull-zfull.detach())
            predicted=forward(value,pe,mask)
            if lossname=="cosine":
                per=1-F.cosine_similarity(predicted[:,1:],h[:,1:],dim=-1)
            else:
                residual=predicted[:,1:]-h[:,1:]
                if lossname=="white":residual=residual@self.transform
                per=residual.square().sum(-1)
            loss=per.mean()
            if not torch.isfinite(loss):raise RuntimeError("nonfinite discrete objective")
            loss.backward()
            if not torch.isfinite(z.grad).all():raise RuntimeError("nonfinite discrete gradient")
            val=float(loss.detach());trace.append(val)
            if val<bestloss:bestloss=val;bestids=ids.detach().clone()
            changes.append(int((ids!=bestids).sum()))
            opt.step()
        tokens=bestids.cpu();torch.cuda.synchronize();end=time.perf_counter()
        return {"tokens":tokens},{"total":end-start,"loss":trace,"changes_from_best":changes,
          "whole_sequence_forward_backward_evaluations":steps,
          "token_position_forward_evaluations":steps*len(tokens),
          "per_position_shortlist_size":None,"full_vocabulary_projections":steps+1,
          "separate_candidate_verifier_calls":0}


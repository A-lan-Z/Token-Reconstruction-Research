"""Parallel full-vocabulary inversion conditional on one reconstructed context."""
import torch
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
from fast_soft import FastSoftVocabulary,BF16Mixture

CONFIGS=[("control",.6,0.)]+[(f"context_lr{str(lr).replace('.','p')}_g{str(g).replace('.','p')}",lr,g)
 for lr in [.3,.6,1.] for g in [0.,.003]]

def projected_attention(layer,h,pe):
    inp=layer.input_layernorm(h);att=layer.self_attn
    shape=(*inp.shape[:-1],-1,att.head_dim)
    q=att.q_proj(inp).view(shape).transpose(1,2)
    k=att.k_proj(inp).view(shape).transpose(1,2)
    v=att.v_proj(inp).view(shape).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*pe)
    return q,repeat_kv(k,att.num_key_value_groups),repeat_kv(v,att.num_key_value_groups)

def finish_layer(layer,h,attention):
    out=attention.transpose(1,2).reshape(*h.shape).contiguous()
    h=h+layer.self_attn.o_proj(out)
    return h+layer.mlp(layer.post_attention_layernorm(h))

def conditional_forward(prefix,soft,context,pe,mask):
    dtype=prefix.embed_tokens.weight.dtype
    h=soft.to(dtype);hard=context.detach().to(dtype)
    for layer in prefix.layers:
        with torch.no_grad():
            hq,hk,hv=projected_attention(layer,hard,pe)
            hs=(hq@hk.transpose(-1,-2))*layer.self_attn.scaling
            if mask is not None:hs=hs+mask
            ha=F.softmax(hs.float(),dim=-1).to(dtype)
            hard=finish_layer(layer,hard,ha@hv)
        q,k,v=projected_attention(layer,h,pe)
        score=(q@hk.transpose(-1,-2))*layer.self_attn.scaling
        own=(q*k).sum(-1)*layer.self_attn.scaling
        score=score-torch.diag_embed(score.diagonal(dim1=-2,dim2=-1))+torch.diag_embed(own)
        if mask is not None:score=score+mask
        a=F.softmax(score.float(),dim=-1).to(dtype)
        out=a@hv+a.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*(v-hv)
        h=finish_layer(layer,h,out)
    return h.float()

def reference_tests():
    from transformers import LlamaConfig,LlamaForCausalLM
    from token_reconstruction.public_prefix import ContiguousPublicPrefix
    torch.manual_seed(200029)
    cfg=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=16)
    cfg._attn_implementation="eager"
    prefix=ContiguousPublicPrefix(LlamaForCausalLM(cfg),2).eval().requires_grad_(False)
    hard=torch.randn(1,5,32)
    soft=torch.randn(1,5,32,requires_grad=True)
    pe=prefix.rotary_emb(soft,torch.arange(5).view(1,-1))
    mask=prefix._causal_mask(soft,start_pos=0,total_tokens=5)
    output=conditional_forward(prefix,soft,hard,pe,mask)
    forward_error=0.;gradient_error=0.
    probe=torch.randn(32)
    for i in range(5):
        z=hard.clone();z[:,i]=soft[:,i]
        for layer in prefix.layers:z=prefix._hidden(layer(z,position_embeddings=pe,attention_mask=mask,use_cache=False))
        ref=z[:,i];got=output[:,i]
        torch.testing.assert_close(got,ref,rtol=1e-5,atol=1e-6)
        go=torch.autograd.grad((got*probe).sum(),soft,retain_graph=True)[0]
        gr=torch.autograd.grad((ref*probe).sum(),soft,retain_graph=True)[0]
        torch.testing.assert_close(go,gr,rtol=1e-4,atol=1e-6)
        assert torch.count_nonzero(go[:,[j for j in range(5) if j!=i]])==0
        forward_error=max(forward_error,float((got-ref).detach().abs().max()))
        gradient_error=max(gradient_error,float((go-gr).abs().max()))
    return {"forward_max_error":forward_error,"gradient_max_error":gradient_error,"independent_single_position_references":5,"other_position_gradients_zero":True}

class HardContextSoft(FastSoftVocabulary):
    def __init__(self,prefix,lr,gini):
        super().__init__(prefix,"adaptive",128)
        self.base_lr=lr;self.gini=gini
    def evaluate(self,length):
        pe,mask=self.geometry[length]
        probability=F.softmax(self.logits[:length-1],dim=-1)
        soft=BF16Mixture.apply(probability,self.weight)
        ids=self.logits[:length-1].argmax(-1)
        context=torch.cat([self.bos,self.weight[ids][None].float()],dim=1)
        predicted=conditional_forward(self.prefix,torch.cat([self.bos,soft[None]],dim=1),context,pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],self.h[None,1:length],dim=-1))[0]
        loss=error.mean()
        if self.gini:loss=loss+self.gini*(self.counter[0]>=32)*(1-probability.square().sum(-1)).mean()
        with torch.no_grad():
            self.tokens[1:length].copy_(ids)
            eligible=(self.counter[0]>=32) if self.gini else torch.ones((),device="cuda",dtype=torch.bool)
            improved=(loss.detach()<self.best_loss)&eligible
            self.best_tokens[1:length].copy_(torch.where(improved,ids,self.best_tokens[1:length]))
            self.best_loss.copy_(torch.where(eligible,torch.minimum(self.best_loss,loss.detach()),self.best_loss))
            per=error.detach();better=(per<self.position_best_loss[:length-1])&eligible
            self.position_best_tokens[1:length].copy_(torch.where(better,ids,self.position_best_tokens[1:length]))
            self.position_best_loss[:length-1].copy_(torch.where(eligible,torch.minimum(self.position_best_loss[:length-1],per),self.position_best_loss[:length-1]))
            self.position_error[:length-1].copy_(per)
            self.loss_trace.scatter_(0,self.counter,loss.detach().reshape(1));self.counter.add_(1)
        return loss
    @torch.no_grad()
    def reset(self,h):
        super().reset(h);self.lr_tensor.fill_(self.base_lr)
    def decode(self,observations):
        out,stats=super().decode(observations)
        stats.update(base_lr=self.base_lr,gini=self.gini,context="current_argmax_tokens_detached",
          additional_context_prefix_forwards=self.steps+1,conditional_soft_forwards=self.steps+1,
          gradient="own_position_only_with_hard_reconstructed_past",attention="explicit_eager_new_numerical_variant")
        return out,stats

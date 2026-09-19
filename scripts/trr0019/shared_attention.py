"""A2 attention reads one shared past plus each candidate's current K/V.

Mathematically the native single-query attention. Floating-point implementation
is different: qualification and full reconstruction evaluation are required.
"""
import torch
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

@triton.jit
def attention_kernel(Q,K,V,PK,PV,O,T:tl.constexpr,H:tl.constexpr,KV:tl.constexpr,D:tl.constexpr,BT:tl.constexpr):
    c=tl.program_id(0);head=tl.program_id(1);kh=head//(H//KV)
    d=tl.arange(0,D);t=tl.arange(0,BT)
    q=tl.load(Q+c*H*D+head*D+d).to(tl.float32)
    kp=tl.load(PK+kh*T*D+t[:,None]*D+d[None,:],t[:,None]<T,0).to(tl.float32)
    vp=tl.load(PV+kh*T*D+t[:,None]*D+d[None,:],t[:,None]<T,0).to(tl.float32)
    kc=tl.load(K+c*KV*D+kh*D+d).to(tl.float32)
    vc=tl.load(V+c*KV*D+kh*D+d).to(tl.float32)
    keys=tl.where(t[:,None]==T,kc[None,:],kp)
    values=tl.where(t[:,None]==T,vc[None,:],vp)
    logits=tl.sum(keys*q[None,:],1)*(D**-0.5)
    logits=tl.where(t<=T,logits,float('-inf'))
    weights=tl.exp(logits-tl.max(logits,0));weights=weights/tl.sum(weights,0)
    out=tl.sum(weights[:,None]*values,0)
    tl.store(O+c*H*D+head*D+d,out)

def shared_attention(q,k,v,pk,pv):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert pk.is_contiguous() and pv.is_contiguous()
    c,h,_,d=q.shape;kv=k.shape[1];t=pk.shape[2]
    out=torch.empty((c,1,h,d),device=q.device,dtype=q.dtype)
    attention_kernel[(c,h)](q,k,v,pk,pv,out,t,h,kv,d,triton.next_power_of_2(t+1),num_warps=4,enable_fp_fusion=False)
    return out

@torch.inference_mode()
def fast_candidates(prefix,cache,ids,pos):
    if pos!=cache.length or len(ids)!=256:raise ValueError('qualified K256 only')
    hidden=prefix.embed_tokens(ids.reshape(-1,1))
    positions=torch.arange(pos,pos+1,device=hidden.device).view(1,1).expand(len(ids),-1)
    cos,sin=prefix.rotary_emb(hidden,positions)
    for i,layer in enumerate(prefix.layers):
        residual=hidden;norm=layer.input_layernorm(hidden);a=layer.self_attn
        shape=(*norm.shape[:-1],-1,a.head_dim)
        q=a.q_proj(norm).view(shape).transpose(1,2)
        k=a.k_proj(norm).view(shape).transpose(1,2)
        v=a.v_proj(norm).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        past=cache.backend.layers[i]
        attended=shared_attention(q,k,v,past.keys,past.values)
        hidden=residual+a.o_proj(attended.reshape(len(ids),1,-1))
        hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden[:,-1].float()

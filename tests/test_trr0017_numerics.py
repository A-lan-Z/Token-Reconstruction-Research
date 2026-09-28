"""Independent numerical contracts for the new execution/gradient code."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/trr0017"))
from shared_context import SharedContext
from discrete_parallel import DiscreteParallel
from token_reconstruction.public_prefix import ContiguousPublicPrefix

def test_shared_context_is_native_materialization_and_does_not_modify_source():
    torch.manual_seed(17)
    k=torch.randn(1,2,4,8);v=torch.randn(1,2,4,8)
    source=SimpleNamespace(length=4,backend=SimpleNamespace(layers=[SimpleNamespace(keys=k,values=v)]))
    before=(k.clone(),v.clone());cache=SharedContext(source,3,1)
    qk=torch.randn(3,2,1,8);qv=torch.randn(3,2,1,8)
    a,b=cache.update(qk,qv,0)
    assert torch.equal(a,torch.cat([k.repeat_interleave(3,0),qk],-2))
    assert torch.equal(b,torch.cat([v.repeat_interleave(3,0),qv],-2))
    assert a.is_contiguous() and b.is_contiguous()
    assert torch.equal(k,before[0]) and torch.equal(v,before[1])
    assert source.length==4 and cache.get_seq_length(0)==5
    with pytest.raises(ValueError,match="once"):cache.update(qk,qv,0)
    with pytest.raises(ValueError,match="shape"):
        SharedContext(source,3,1).update(qk[:2],qv[:2],0)

def test_diagonal_gradient_matches_own_position_native_jacobian():
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(1700);torch.set_num_threads(2)
    cfg=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                   num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=16)
    cfg._attn_implementation="eager"
    prefix=ContiguousPublicPrefix(LlamaForCausalLM(cfg),2).to(torch.bfloat16).eval()
    prefix.requires_grad_(False)
    engine=object.__new__(DiscreteParallel);engine.prefix=prefix
    z=torch.randn(1,4,32,requires_grad=True)
    pos=torch.arange(4).view(1,-1);pe=prefix.rotary_emb(z.to(torch.bfloat16),pos)
    mask=prefix._causal_mask(z.to(torch.bfloat16),start_pos=0,total_tokens=4)
    diagonal=engine.diagonal_forward(z,pe,mask)
    native=engine.forward(z,pe,mask)
    assert torch.equal(diagonal,native)
    for i in range(4):
        gd=torch.autograd.grad(diagonal[0,i].sum(),z,retain_graph=True)[0]
        gn=torch.autograd.grad(native[0,i].sum(),z,retain_graph=True)[0]
        other=[j for j in range(4) if j!=i]
        assert torch.count_nonzero(gd[:,other])==0
        torch.testing.assert_close(gd[0,i],gn[0,i],rtol=.02,atol=.02)
    assert torch.count_nonzero(torch.autograd.grad(native[0,-1].sum(),z)[0][:,:-1])>0


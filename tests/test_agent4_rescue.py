"""Cache correctness and final-return safety, independent tiny public fixtures."""
import sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/agent4_rescue'))
from executor import CurrentToken
from transformers import LlamaConfig,LlamaForCausalLM
from token_reconstruction.public_prefix import ContiguousPublicPrefix
from solver import continuous_forward
from cached import reconstruct,Settings

def model():
    torch.manual_seed(910)
    c=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1)
    c._attn_implementation='eager'
    return ContiguousPublicPrefix(LlamaForCausalLM(c),2).eval().requires_grad_(False)

def test_trials_keep_committed_cache_and_gradient():
    p=model();ex=CurrentToken(p,bos=1);ex.commit(4)
    before={i:(k.clone(),v.clone()) for i,(k,v) in ex.state.items()}
    z=p.embed_tokens.weight[7].detach().clone().requires_grad_()
    y=ex(z);g,=torch.autograd.grad(y.square().mean(),z)
    zz=z.detach().clone().requires_grad_()
    full=continuous_forward(p,torch.cat((p.embed_tokens(torch.tensor([[1,4]])).detach(),zz.view(1,1,-1)),1))[0,-1]
    gg,=torch.autograd.grad(full.square().mean(),zz)
    assert torch.allclose(y,full,atol=1e-6,rtol=1e-5)
    assert torch.allclose(g,gg,atol=1e-6,rtol=1e-5)
    assert ex.length==2 and all(torch.equal(k,before[i][0]) and torch.equal(v,before[i][1]) for i,(k,v) in ex.state.items())
    ex.commit(7)
    assert ex.length==3 and all(k.shape[-2]==3 for k,v in ex.state.values())

def test_changed_weights_invalidate_cached_state():
    p=model();ex=CurrentToken(p,bos=1)
    with torch.no_grad():p.embed_tokens.weight.add_(.001)
    with pytest.raises(RuntimeError,match='weights changed'):ex(p.embed_tokens.weight[4])
    ex.invalidate();ex.commit(1)
    assert ex.length==1

def test_cached_exhaustion_returns_best_evaluated_and_preserves_weights():
    p=model();before={k:v.clone() for k,v in p.state_dict().items()}
    r=reconstruct(p,torch.full((3,16),100.),Settings(steps=3,optimizer='adam',learning_rate=.01),bos=1)
    for t in r['trace']:
        assert t['token']==min(t['checks'],key=lambda c:c['mse'])['token']
        assert len(t['checks'])==3
    assert all(torch.equal(v,before[k]) for k,v in p.state_dict().items())
    assert all(p.grad is None for p in p.parameters())

def test_cached_timeout_keeps_whole_denominator():
    p=model();r=reconstruct(p,torch.zeros((4,16)),Settings(record_seconds=0),bos=1)
    assert r['tokens']==[1,-1,-1,-1]

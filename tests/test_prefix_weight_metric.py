import types
import pytest
import torch
from torch import nn
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric

def fixture():
    torch.manual_seed(17)
    p=nn.Module();p.embed_tokens=nn.Embedding(11,4)
    layer=nn.Module();layer.self_attn=nn.Module();layer.self_attn.o_proj=nn.Linear(4,4,bias=False)
    layer.mlp=nn.Module();layer.mlp.down_proj=nn.Linear(7,4,bias=False)
    p.layers=nn.ModuleList([layer]);p.requires_grad_(False)
    return p

def test_metric_is_same_as_explicit_mahalanobis():
    p=fixture();m=PrefixWeightMetric(p);m.build()
    C=sum(W@W.T/(W@W.T).trace() for W in [p.layers[0].self_attn.o_proj.weight,p.layers[0].mlp.down_proj.weight])
    inv=torch.linalg.inv(C);E=p.embed_tokens.weight;h=torch.randn(5,4)
    score=(h@inv@E.T)/torch.sqrt((E@inv*E).sum(-1))[None,:]
    assert torch.equal(m.propose(h,5),score.topk(5,dim=-1).indices)
    assert not any(x.requires_grad for x in [m.table,m.transform])

def test_cache_rejects_inplace_prefix_change():
    p=fixture();m=PrefixWeightMetric(p);m.build()
    with torch.no_grad():p.layers[0].self_attn.o_proj.weight.add_(.1)
    with pytest.raises(RuntimeError,match="prefix changed"):m.propose(torch.zeros(1,4))

def test_cache_rejects_replaced_parameter():
    p=fixture();m=PrefixWeightMetric(p);m.build()
    p.embed_tokens.weight=nn.Parameter(p.embed_tokens.weight.clone(),requires_grad=False)
    with pytest.raises(RuntimeError,match="prefix changed"):m.propose(torch.zeros(1,4))

def test_bad_observation_fails_closed():
    p=fixture();m=PrefixWeightMetric(p);m.build()
    with pytest.raises(ValueError):m.propose(torch.full((1,4),float("nan")),2)

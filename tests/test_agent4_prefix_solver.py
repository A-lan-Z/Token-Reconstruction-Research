"""Scientific invariants for the task-local bounded search, using tiny fixtures."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from token_reconstruction.public_prefix import ContiguousPublicPrefix

path = Path(__file__).resolve().parents[1] / 'scripts/agent4/solver.py'
spec = importlib.util.spec_from_file_location('agent4_solver_tested', path)
solver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = solver
spec.loader.exec_module(solver)


def fixture():
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1)
    config._attn_implementation = 'eager'
    return ContiguousPublicPrefix(LlamaForCausalLM(config), 1).eval().requires_grad_(False)


def test_exhaustion_emits_only_a_verified_candidate_and_never_changes_weights():
    prefix = fixture()
    before = {k: v.clone() for k, v in prefix.state_dict().items()}
    observed = torch.full((3, 8), 100.0)  # deliberately not an exact match
    result = solver.reconstruct(prefix, observed, solver.Settings(steps=3), bos=1)
    for trace in result['trace']:
        assert trace['reason'] == 'budget_exhausted'
        assert len(trace['checks']) == 3
        assert trace['gradient_steps'] == 3
        assert trace['token'] == min(trace['checks'], key=lambda c: c['mse'])['token']
        assert len({c['token'] for c in trace['checks']}) == 3
    assert all(torch.equal(before[k], v) for k, v in prefix.state_dict().items())
    assert all(p.grad is None for p in prefix.parameters())


def test_one_check_cannot_return_the_unverified_post_gradient_projection():
    prefix = fixture()
    result = solver.reconstruct(prefix, torch.ones((2, 8)), solver.Settings(steps=1), bos=1)
    trace = result['trace'][0]
    assert trace['token'] == trace['initial_token'] == trace['checks'][0]['token']
    assert len(result['tokens']) == 2


def test_invalid_observation_or_trainable_prefix_fails_closed():
    prefix = fixture()
    with pytest.raises(ValueError, match='finite'):
        solver.reconstruct(prefix, torch.full((2, 8), float('nan')), bos=1)
    prefix.embed_tokens.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        solver.reconstruct(prefix, torch.zeros((2, 8)), bos=1)


def test_record_timeout_retains_the_whole_unreconstructed_suffix():
    prefix = fixture()
    result = solver.reconstruct(prefix, torch.zeros((4, 8)),
                                solver.Settings(record_seconds=0), bos=1)
    assert result['tokens'] == [1, -1, -1, -1]
    assert result['trace'] == []

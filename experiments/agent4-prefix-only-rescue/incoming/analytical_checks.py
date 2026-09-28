"""Analytical/synthetic checks for a prefix-only inversion research proposal.

These tests use no Llama weights, no target observations, and no private data.
They do not reproduce Agent 4's 29/30 result or estimate real inference speed.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(1)
DTYPE = torch.float64

def loss_scaling_check() -> dict:
    d = 2048
    contraction = 1.0 - 2.0 / d
    # Identity map F(z)=z, loss=mean((z-h)**2), SGD learning rate 1,
    # without clipping/schedule/projection. Closed form, not a SIPIT replay.
    return {
        'definition': 'identity forward map, coordinate-mean squared loss, SGD lr=1',
        'dimension': d,
        'error_multiplier_per_step': contraction,
        'error_fraction_after_50_steps': contraction ** 50,
        'error_fraction_after_256_steps': contraction ** 256,
        'error_fraction_after_1000_steps': contraction ** 1000,
        'steps_for_100fold_error_norm_reduction': math.ceil(math.log(0.01) / math.log(contraction)),
        'curvature_scaled_identity_step_lr': d / 2,
        'curvature_scaled_steps_for_exact_solution_in_exact_arithmetic': 1,
    }

def embedding_distance_check() -> dict:
    rng = np.random.default_rng(2048)
    E = rng.normal(size=(257, 32))
    z = rng.normal(size=(32,))
    explicit = np.sum((E-z)**2, axis=1)
    reduced = np.sum(E*E, axis=1) - 2 * (E @ z) + z @ z
    assert np.allclose(explicit, reduced, rtol=1e-12, atol=1e-12)
    assert np.array_equal(np.argsort(explicit), np.argsort(reduced))
    vocab, width = 128256, 2048
    return {
        'synthetic_vocabulary': 257,
        'max_float64_distance_difference': float(np.max(np.abs(explicit-reduced))),
        'full_ranking_equal': True,
        'real_geometry_vocabulary': vocab,
        'real_geometry_width': width,
        'one_fp32_vocabulary_matrix_bytes': vocab*width*4,
        'one_fp32_vocabulary_matrix_gib': vocab*width*4/2**30,
        'one_fp32_score_vector_bytes': vocab*4,
        'one_bool_visited_vector_bytes': vocab,
        'qualification': 'matrix-vector form still reads the full table; it avoids its V-by-d difference temporary and clone, not its FLOPs or bandwidth',
    }

def local_geometry_check() -> dict:
    J = np.diag([100.0, 1.0])
    z = np.array([0.08, 0.01])
    gold = np.array([0.0, 1.0])
    wrong = np.array([0.08, 0.30])
    E = np.stack([gold, wrong])
    h = J @ gold
    r = J @ z - h
    embedding_dist = np.linalg.norm(E-z, axis=1)
    local_forward_dist = np.linalg.norm(r + (E-z) @ J.T, axis=1)
    exact_forward_dist = np.linalg.norm(E @ J.T-h, axis=1)
    assert embedding_dist.argmin() == 1
    assert local_forward_dist.argmin() == 0
    assert np.allclose(local_forward_dist, exact_forward_dist)
    step = np.linalg.solve(J.T @ J, -(J.T @ r))
    assert np.allclose(z+step, gold)
    return {
        'definition': 'deliberately anisotropic two-dimensional linear counterexample, not language-model evidence',
        'embedding_distance_gold_wrong': embedding_dist.tolist(),
        'linearized_boundary_distance_gold_wrong': local_forward_dist.tolist(),
        'nearest_embedding_is_wrong': True,
        'local_boundary_ranking_is_correct': True,
        'gauss_newton_solution': (z+step).tolist(),
    }

def discrete_surrogate_check() -> dict:
    rng=np.random.default_rng(64)
    E=rng.normal(size=(193,16))
    z=rng.normal(size=16)
    g=rng.normal(size=16)
    lam=3.5
    shifts=E-z
    direct=shifts@g+0.5*lam*np.sum(shifts**2,axis=1)
    efficient=E@(g-lam*z)+0.5*lam*np.sum(E**2,axis=1)
    centered_difference=(direct-direct[0])-(efficient-efficient[0])
    assert np.max(np.abs(centered_difference))<1e-11
    assert np.array_equal(np.argsort(direct),np.argsort(efficient))
    return {
        'definition':'first-order discrete loss surrogate plus quadratic distance penalty',
        'max_centered_difference':float(np.max(np.abs(centered_difference))),
        'full_ranking_equal':True,
        'qualification':'same surrogate ranking in exact arithmetic; not a guarantee the nonlinear real loss follows that ranking',
    }

class Block(nn.Module):
    def __init__(self,d:int)->None:
        super().__init__()
        self.q=nn.Linear(d,d,bias=False,dtype=DTYPE)
        self.k=nn.Linear(d,d,bias=False,dtype=DTYPE)
        self.v=nn.Linear(d,d,bias=False,dtype=DTYPE)
        self.o=nn.Linear(d,d,bias=False,dtype=DTYPE)
        self.down=nn.Linear(d,2*d,dtype=DTYPE)
        self.up=nn.Linear(2*d,d,dtype=DTYPE)
        self.d=d
    def full(self,x:torch.Tensor):
        y=F.layer_norm(x,(self.d,))
        q,k,v=self.q(y),self.k(y),self.v(y)
        s=q@k.T/math.sqrt(self.d)
        mask=torch.triu(torch.ones_like(s,dtype=torch.bool),diagonal=1)
        a=x+self.o(torch.softmax(s.masked_fill(mask,-torch.inf),dim=-1)@v)
        out=a+self.up(F.gelu(self.down(F.layer_norm(a,(self.d,)))))
        return out,(k.detach().clone(),v.detach().clone())
    def one(self,x:torch.Tensor,cache):
        y=F.layer_norm(x,(self.d,))
        q,k,v=self.q(y),self.k(y),self.v(y)
        K=torch.cat([cache[0],k],dim=0)
        V=torch.cat([cache[1],v],dim=0)
        a=x+self.o(torch.softmax(q@K.T/math.sqrt(self.d),dim=-1)@V)
        return a+self.up(F.gelu(self.down(F.layer_norm(a,(self.d,)))))

def cached_gradient_check()->dict:
    torch.manual_seed(620)
    blocks=nn.ModuleList([Block(16),Block(16)]).eval()
    blocks.requires_grad_(False)
    output_errors=[]; gradient_errors=[]
    for length in [1,4,16]:
        prefix=torch.randn(length,16,dtype=DTYPE)
        current=torch.randn(1,16,dtype=DTYPE)
        target=torch.randn(1,16,dtype=DTYPE)
        with torch.no_grad():
            cache=[]
            work=prefix
            for layer in blocks:
                work,kv=layer.full(work)
                cache.append(kv)
        before=[(k.clone(),v.clone()) for k,v in cache]
        z1=current.clone().requires_grad_(True)
        all_x=torch.cat([prefix,z1],dim=0)
        for layer in blocks:
            all_x,_=layer.full(all_x)
        full_y=all_x[-1:]
        full_g=torch.autograd.grad(0.5*torch.sum((full_y-target)**2),z1)[0]
        z2=current.clone().requires_grad_(True)
        y=z2
        for layer,kv in zip(blocks,cache):
            y=layer.one(y,kv)
        cached_g=torch.autograd.grad(0.5*torch.sum((y-target)**2),z2)[0]
        output_errors.append(float((full_y-y).abs().max().detach()))
        gradient_errors.append(float((full_g-cached_g).abs().max()))
        assert torch.allclose(full_y,y,atol=2e-12,rtol=2e-12)
        assert torch.allclose(full_g,cached_g,atol=2e-12,rtol=2e-12)
        assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(before,cache))
    return {
        'definition':'synthetic two-block causal attention prefix, fixed weights, float64, no rotary or grouped-query attention',
        'prefix_lengths':[1,4,16],
        'max_output_absolute_error':max(output_errors),
        'max_input_gradient_absolute_error':max(gradient_errors),
        'prefix_cache_unchanged':True,
        'qualification':'does not validate Agent 4 or Llama cache/rotary/precision integration',
    }

def mismatch_check()->dict:
    # Public states are 0 and 1, but target emits .01 for true token 0.
    states=np.array([0.0,1.0]); observed=0.01
    residual=np.abs(states-observed)
    accepted=np.isclose(states,observed,rtol=1e-5,atol=1e-5)
    assert not accepted.any() and residual.argmin()==0
    return {
        'definition':'two-token public identity map with small target discrepancy',
        'candidate_residuals':residual.tolist(),
        'strict_verifier_accepts':accepted.tolist(),
        'best_candidate_is_true':True,
        'sufficient_margin_check':float(abs(observed)) < 0.5,
        'qualification':'rejecting a strict equality match does not justify discarding the best token under mismatch',
    }

def main()->None:
    output={
        'scope':'Analytical and synthetic checks only. No Llama model run; no measured rescue speedup.',
        'loss_scaling':loss_scaling_check(),
        'embedding_distance':embedding_distance_check(),
        'local_geometry':local_geometry_check(),
        'discrete_surrogate':discrete_surrogate_check(),
        'cached_gradient':cached_gradient_check(),
        'mismatch_acceptance':mismatch_check(),
        'pilot_arithmetic': {
            'source': 'User-supplied Agent 4 summary, not independently remeasured',
            'prefix_only_seconds': 9.82,
            'comparison_seconds': 1.69,
            'time_ratio': 9.82 / 1.69,
            'fraction_of_current_time_to_eliminate_to_tie': 1 - 1.69 / 9.82,
            'qualifier': 'Arithmetic at the reported scope, not a measured bottleneck decomposition.',
        },
    }
    path=Path(__file__).with_name('analytical_results.json')
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    main()

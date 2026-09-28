"""Arithmetic and toy counterexamples only; no Llama experiments or private data.
Run: python analytical_checks.py
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np


def headroom(total: float, reference: float, fraction: float) -> dict:
    """Amdahl-style conditional projection, not an observed runtime."""
    remainder = reference / total - (1.0 - fraction)
    required = fraction / remainder if remainder > 0 else None
    return {
        'modifiable_fraction': fraction,
        'best_possible_seconds_if_part_free': total * (1 - fraction),
        'required_speedup_of_part_to_tie': required,
        'projected_seconds_if_part_10x_faster': total * ((1 - fraction) + fraction/10),
    }


def main() -> None:
    n, old_correct, new_correct, ref_correct = 92, 77, 85, 92
    old_time, new_time, ref_time = 117.67, 22.84, 5.20
    result = {
        'scope': 'User-reported aggregate arithmetic and constructed examples; not new model measurements.',
        'reported_aggregate_calculations': {
            'old_token_accuracy_percent': 100*old_correct/n,
            'new_token_accuracy_percent': 100*new_correct/n,
            'reference_token_accuracy_percent': 100*ref_correct/n,
            'speedup_over_old': old_time/new_time,
            'relative_to_reference': new_time/ref_time,
            'time_reduction_to_tie_percent': 100*(1-ref_time/new_time),
            'token_error_reduction_percent': 100*((n-old_correct)-(n-new_correct))/(n-old_correct),
            'warning': 'Numbers of source clips, paired error locations, and timing distributions were not supplied. No binomial token CI is appropriate.',
        },
        'conditional_headroom': [headroom(new_time, ref_time, f) for f in [.4,.6,.8,.9]],
    }
    # Two explicitly invented per-position profiles with the identical aggregate.
    concentrated = np.r_[np.full(85, .02), np.full(7, 3.02)]
    diffuse = np.full(92, new_time/92)
    assert np.isclose(concentrated.sum(), new_time)
    assert np.isclose(diffuse.sum(), new_time)
    result['synthetic_cost_profiles'] = {
        'concentrated_total': float(concentrated.sum()),
        'concentrated_largest_seven_fraction': float(np.sort(concentrated)[-7:].sum()/new_time),
        'concentrated_total_if_seven_reduced_to_point_one': float(85*.02 + 7*.1),
        'diffuse_total': float(diffuse.sum()),
        'diffuse_largest_seven_fraction': float(np.sort(diffuse)[-7:].sum()/new_time),
        'diffuse_total_even_if_seven_free': float(np.sort(diffuse)[:-7].sum()),
        'warning': 'Neither profile is claimed to describe Agent 4. They show aggregate non-identifiability.',
    }
    # Causal toy model: F(x1,x2)=(x1,2*x1+x2), vocabulary {0,1}.
    # Unknown true sequence (0,1). Observed first coordinate perturbed by +.55.
    truth = np.array([0., 1.])
    A = np.array([[1.,0.],[2.,1.]])
    observation = A @ truth + np.array([.55,0.])
    x1 = min([0.,1.], key=lambda x:(x-observation[0])**2)
    x2 = min([0.,1.], key=lambda x:(2*x1+x-observation[1])**2)
    greedy = np.array([x1,x2])
    candidates = [np.array(x,float) for x in itertools.product([0,1],repeat=2)]
    costs = [float(np.square(A@x-observation).sum()) for x in candidates]
    winner = candidates[int(np.argmin(costs))]
    continuous = np.linalg.solve(A,observation)
    rounded = (continuous >= .5).astype(int)
    assert np.array_equal(winner, truth)
    assert not np.array_equal(greedy,truth)
    assert not np.array_equal(rounded,truth)
    result['synthetic_block_evidence'] = {
        'forward_matrix': A.tolist(), 'true_tokens': truth.tolist(),
        'observation': observation.tolist(), 'greedy_tokens': greedy.tolist(),
        'joint_discrete_winner': winner.tolist(),
        'joint_candidates_and_squared_errors': [{'tokens':x.tolist(),'loss':c} for x,c in zip(candidates,costs)],
        'continuous_exact_solution': continuous.tolist(),
        'continuous_nearest_tokens': rounded.tolist(),
        'warning': 'Constructed counterexample only. Later unknown tokens can help in discrete block search; unconstrained relaxation can also absorb the evidence incorrectly.',
    }
    # Orthogonal output drift can be large and still have no effect on a finite
    # candidate ranking. Local worst-case norm/margin bounds are conservative.
    ref = np.array([[0.,0.],[1.,0.]])
    y = np.array([0.,100.])
    sq = np.square(ref-y).sum(axis=1)
    result['synthetic_orthogonal_drift'] = {
        'candidate_states':ref.tolist(), 'observed':y.tolist(),
        'squared_errors':sq.tolist(), 'correct_candidate_index':0,
        'winner':int(np.argmin(sq)),
        'distortion_norm_over_candidate_separation':100.,
        'warning':'Failure of a sufficient norm bound is not proof that any decision flipped.',
    }
    # Projecting onto the span of candidate differences preserves Euclidean
    # rankings exactly; it is not in itself an improved selector.
    rng = np.random.default_rng(7)
    candidate = rng.normal(size=(4,8)); y = rng.normal(size=8)
    _, _, vt = np.linalg.svd(candidate[1:]-candidate[0],full_matrices=False)
    Q = vt.T; projected = (candidate-y) @ Q
    full_d = np.square(candidate-y).sum(axis=1)
    proj_d = np.square(projected).sum(axis=1)
    assert np.allclose(full_d-full_d[0], proj_d-proj_d[0],atol=1e-12)
    result['projection_ranking_identity'] = {
        'max_error_in_pairwise_squared_distance_differences':float(np.max(np.abs((full_d-full_d[0])-(proj_d-proj_d[0])))),
        'meaning':'Removing directions shared by all candidate differences does not alone change exact L2 ordering.',
    }
    out=Path(__file__).with_name('analytical_results.json')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    main()

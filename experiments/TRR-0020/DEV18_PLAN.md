# Development18: current-residual rank-one full-vocabulary loss

This prospective grid follows the fully scored development17 matrix. No active canonical method is added. All development records have been opened historically; results are retrospective and cannot establish fresh confirmation.

## Hypothesis and exact rule
The fixed random sensitivity sketch in development17 did not improve the warm128 result. This experiment recomputes the most directly relevant output-error direction at every whole-sequence update. It uses no shortlist or separate verifier, no model parameter fitting, and only the supplied frozen prefix, observed cut activation and public weights/tokenizer.

Let normalized prefix output be u(z), normalized observation v, residual r=u-v, L=.5||r||^2=1-cos(u,v), and g=J^T r. The Gauss-Newton contribution of the unit residual direction a=r/||r|| is J^T a a^T J=g g^T/(2L). The rest of the Gauss-Newton matrix is positive semidefinite, but this rank-one term alone is not a bound on the nonlinear loss. Its usefulness remains unproven.

For each current hard embedding z and EVERY vocabulary embedding e, set delta=e-z, p=g.delta, M=R R^T (raw or the existing white metric), and lambda=clamp(beta*||g R^{-T}||^2/(2*max(L,1e-8)),1e-6,1e6). Choose the global vocabulary argmax of -p-gamma*p^2/(4*max(L,1e-8))-.5*lambda*||delta R||^2. This adds the current-residual curvature to the existing gradient-scaled isotropic penalty. The score uses two FP32 vocabulary matrix products with TF32 disabled, versus one product in the earlier direct method. Distances calculated from norms and dot products are clamped to zero for roundoff. Each update refreshes g,L and lambda; no random sensitivity probes or per-input quadratic vocabulary cache are used.

The own-position derivative and BF16 prefix arithmetic remain the declared diagonal-forward approximation. Both stages operate on the whole native-length sequence. A128-step or64-step unchanged Gini003 soft start is followed by32 hard full-vocabulary updates. Save steps0/1/2/4/8/16/32, lowest whole observed objective, and lowest observed error per position. The per-position output may combine states evaluated under different reconstructed contexts; no final verification is inserted.

## Fixed matrix and controls
24configurations: warm64/128 x raw/white x beta(.01,.03,.1) x gamma(0,1). Eight existing development observations per configuration yield192cells. Gamma0 is the FP32 arithmetic control for removing the new rank-one term; it is not claimed byte-equivalent to development15's BF16 vocabulary products. Qualify index23 (warm128,white,beta.1,gamma1) first, then the remaining23isolated workers. All native40/128geometries repeat three times on fixed public fixture seed200041. The warm-stage outputs and all four trace prefixes must reproduce archived dev7 in every cell.

Independent validation includes tiny CPU-float64 dense-matrix score identities (seed200044), the normalized-residual rank identity, exact eager-versus-replay output and loss-trace checks, and GPU scores sampled only from public qualification runs against separately calculated CPU-float64 delta losses (rtol2e-4,atol5e-6). Save all qualification outputs before evaluating gates. All scores must be finite. No source labels until every192cell output is frozen and the unchanged complete-matrix gate plus three negative gate checks pass.

## Costs and resource preflight
Reconstruction per input: warm+34whole-sequence prefix forwards, warm+32backwards, and64direct vocabulary products after the warm stage. No sensitivity-only calls. All captures, initialization, synchronization, output copies and scalar numeric checks are charged to their stage; table construction and FP32 embedding-copy preparation are separately timed. The complete outer decode time includes all stage bookkeeping.

Largest geometry is127x128256FP32 scores (62.14MiB each) with2048hidden coordinates, two owned CUDA streams and two cached native geometries per stage. Retain the1002MiB FP32 embedding copy. Using development17's4.850GiB measured peak, removing its70MiB probe/cache storage and allowing roughly.6GiB additional live score temporaries gives a conservative5.5GiB estimate. The isolated process cap remains6GiB; the largest public qualification must pass before the larger matrix. The guard requires GPU free>=2GiB, host available>=8GiB, RSS<=10GiB and temperature<80C. Live preflight is recorded in dev18_preflight.json. Expected matrix<=1000seconds, hard guard timeout1800seconds. Do not alter numerical batching to bypass a resource failure; preserve and exclude the attempt first.

An alternative would add a residual-aligned vector to development17's frozen random sketch. It is deferred because that still needs extra prefix probes and leaves curvature stale after the start. The current rule obtains its directional term from the gradient already required for each update.

A development winner must be registered and evaluated with contemporaneous baseline controls in BOTH canonical setups, completing the full active matrix before any replacement claim. Neither this hypothesis nor development17 proves that the objective is impossible.

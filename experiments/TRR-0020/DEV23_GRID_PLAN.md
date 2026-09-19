# Development23 reconstruction grid: adjoint least-squares corrections

## Evidence and scope
The CPU transpose/regularized-solve references passed. Six actual-prefix adjoint checks passed within4.25e-7maximum absolute error. In the public diagnostic, CGLS8left.339/.355mean relative linear residual at128/40positions, compared with.829/.842for Krylov16. The cost is8forward+9transpose products versus16forward products. On this public fixture CGLS8has similar eager direction runtime and a half step reduces actual prefix mismatch to.524/.519, versus.862/.870for the Krylov16half step. These are local public numerical results, not token-reconstruction performance.

This finite grid now tests reconstruction accuracy. It remains exploratory and retrospective, with no active canonical registration until one rule is selected. All128256vocabulary entries remain active. There is no shortlist, candidate pruning, candidate verifier, offline predictor fitting or update of prefix weights.

## Fixed method and matrix
Use the exact development21continuous engine, full-vocabulary initialization, BF16 warm mixture, FP32 prefix forward, median-embedding-norm direction cap, native geometry and fixed checkpoints4/8/16. Replace the local Krylov direction with the qualified CGLS recurrence, and fix nonlinear step damping.5. Do not project intermediate continuous states to tokens. Readouts remain rawL2/rawcosine/whiteL2/whitecosine over the complete vocabulary; tokens are emitted directly and never forwarded for verification or used to restrict subsequent updates.

Grid: warm{0,32,64} x CGLSiterations{4,8} x ridge multiplier{0,1e-4}, twelve configurations, each with16outer iterations. The ridge per position is multiplier*||J^T rhs||^2/||rhs||^2and stays fixed during its linear solve. BOS direction is always zero. Scoring uses every declared readout and checkpoint, with no source-derived state selection.

Run all12configurations on the existing fixed8development records:96method/input cells,1152checkpoint/readout variants. Preserve warm snapshots and trace prefixes against the original Gini003archive (96anchors); warm32and64initial mixtures must match the corresponding development19fitted embeddings (64anchors). Save all outputs before opening labels, validate full completeness and reject tampered/missing evidence. These inputs were historically opened; no fresh-confirmatory claim.

## Qualification and resources
Each configuration runs in its own process, largestwarm64/CGLS8/ridge1e-4first. At public fixture seed200051, qualify128then40positions with three replay repetitions and one eager local-stage run, requiring exact equality of every output and warm trace. Verify the four readout formulas against CPUfloat64. Check the analytic adjoint dot-product identity at the final public continuous state, supplementing the already-passed independent HF autograd checks. Preserve outputs before numerical gates. No batch/precision/resource workaround may silently alter the rule.

The previous full reconstruction engine peaked6.080GiB; the CGLSrecurrence stores fewer basis vectors but adds transpose intermediates. Estimate below6.5GiB, with8GiB process reserved cap and at least3GiB GPU free. Use fresh live admission, no overlapping GPU jobs, external host/temperature guard and1800second timeout. Expected matrix time below900seconds. Qualify the largest case before running the complete matrix.

Count warm preparation/forward/backward cost,17nonlinear continuous forwards,16*KJacobian products and16*(K+1)transpose products, every vocabulary product/readout, capture, synchronization and output transfer. Checkpoint times include earlier readout work and are conservative development costs. No comparison with historical timings is a paired runtime claim. A promising frozen rule must complete both canonical setups with current controls and the full active-method matrix.

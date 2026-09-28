# Development25: full causal inverse hypothesis

Development24's normalized diagonal-block inverse did not close the reconstruction gap.
Its linear operators ignore how an earlier token correction changes later keys, values and attention.
Development25 includes these cross-position paths explicitly. The prefix forward, model parameters,
normalization rule and known BOS remain unchanged. There is no shortlist and no separate token verifier.

The derivative uses full dQ K^T + Q dK^T and dA V + A dV attention terms.
The adjoint accumulates all later-position key/value contributions. Input directions and adjoints
are projected to zero at BOS. The coupled least-squares recurrence uses global sequence inner
products, not independent per-position coefficients. A fixed positive ridge and clipped half
nonlinear step are potential reconstruction choices; no reconstruction grid is yet selected.

Qualification sequence:
1. Tiny FP64 dense autograd for raw and normalized full causal J/JT, duality, zero/BOS checks,
   and CGLS against a dense regularized solve. CPU seed200060.
2. Actual supplied FP32 prefix at128/40 positions on archived public development21 fixtures:
   independent HF JVP and VJP checks for broad, early and late directions; exact forward anchor.
3. Public normalized linear diagnostic with8/16 iterations and ridge multipliers1e-4/1e-2,
   full coupled versus old diagonal equations. Three repetitions each; explicit global residual
   and nonlinear half-step mismatch recorded. This is numerical qualification, not token accuracy.
4. Define and freeze a reconstruction grid only if evidence warrants it. Retrospective labels
   remain excluded until complete output freeze. Any selected active rule needs both canonical
   setups and the full method/setup matrix.

Before GPU execution write a resource preflight; largest public128 geometry first, exclusive guard,
FP32 with TF32 off and deterministic CUBLAS. No dense actual-prefix Jacobian is materialized.
No inference advantage or goal completion is implied by a passing derivative reference.

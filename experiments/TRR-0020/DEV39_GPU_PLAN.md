# Development39 actual-prefix qualification and linear diagnostic

The20tiny CPU derivative cases and eight coupled linear solves passed. This
next run remains a public numerical diagnostic, not reconstruction.

Use the supplied public four-layer prefix converted toFP32, native eager
attention, frozen weights and original rotary metadata. Fixed public random
sequences use CPU seed39039+length at lengths128and40, with known BOS.
For each geometry test actual token embeddings and an activation-shaped input:
nativeFP32cut activations with the BOS input reset to its declared embedding.
At strengths1,.75,.5,.25,0, this gives20predeclared cells. Largest128/activation/
strength1runs first.

Compare the homotopy forward against independently scaled native modules:
forward hooks multiply attention o_proj and MLPdown_proj outputs by strength.
Hooks are removed after each reference call and weights are never modified.
Compare analytic input, strength and combined JVPs against native autogradJVP,
using the existingrtol5e-4/atol5e-5qualification tolerance. Input tangent atBOS is
zero; strength tangent may changeBOS intermediate activations. Require exact
identity at0and exact prior custom-forward output at1. Repeat the custom
forward/JVP outputs exactly.

At each of20cells, evaluate the proposed predictor's linear equation
J_unknown d = -dF/dstrength using the global GMRES primitive at4,8,16iterations,
ridge1e-6, three repeated solves. This makes60linear cells/180repetitions.
Report actual residual, direction norm, wall time and work counts; no correctness
threshold selects an iteration budget. Preserve all arrays and failed attempts.
The known public IDs create fixtures only; no token reconstruction is performed.

The derivative checks qualify algebra on actual weights. The linear residuals
and costs will determine whether a bounded predictor/corrector path is worth
testing. They cannot establish a successful nonlinear path or token accuracy.
Full-vocabulary final projection and any prefix-weight refresh cost remain
outstanding. The canonical registry stays unchanged.

# Full-vocabulary mirror update: distinct from development8

Development28 completed36public cases and108repetitions. Static prefix metrics change
nonlinear response only slightly and inconsistently; inverse scaling is usually worse.
Neither is selected for a reconstruction grid. Development27 showed that weak continuous
current-token updates barely change discrete readout despite reducing activation mismatch.

Development8 already tested the power1 softmax-preconditioned gradient with Adam,
periodic moment resets, logit decay.98 and an observed-error learning-rate multiplier.
Do not rename or repeat that as a new idea. The present hypothesis instead uses the direct
KL-regularized linear probability update, with no Adam moments, logit decay or fitted weights.

For full-vocabulary probabilities p=softmax(z) and probability-space loss derivative G,
q is proportional to p exp(-eta G). Compute it in log space, subtracting the row maximum
only for numerical stability. Every vocabulary logit stays finite; no top-K, masking,
candidate proposal or separate verification is introduced. The forward objective remains
the observed cosine error, using the existing BF16 mixture and frozen public prefix.
Probability-space gradients may reuse the already qualified power1 backward, whose
additive row constant has no effect on this mirror update.

Use eta=min(sqrt(2*tau / Var_p(G)),4/(max(G)-min(G))) with explicit zero-span guards.
The variance relation is a local KL approximation; the actual KL is not guaranteed equal
to tau. The cap limits the range of logit changes to4 per step. Three fixed tau values:
.01,.1,1. No claim of improved nonlinear objective follows from the linear proximal proof.

First qualify the CPU update against direct probability arithmetic and KKT conditions,
including zero/constant gradients and finite extreme logits. Then integrate a new optimizer
and independently qualify actual-prefix gradients, eager/replay equality and public128/40
whole decodes before any reconstruction matrix. Proposed reconstruction family: unchanged
Gini warm0/64 x tau.01/.1/1, with64mirror steps and declared checkpoints8/16/32/64, giving
six configurations x eight retrospective inputs =48cells. Freeze source/assets/plan and
all48outputs before scoring; require exact warm and initial-state anchors. Account for
warm and mirror work, setup/capture, I/O and peak memory. Only a promising fixed rule can
be registered for both canonical setups. This plan is a hypothesis, not an efficacy result.

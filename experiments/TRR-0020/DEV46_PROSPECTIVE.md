# Development46 prospective: remove duplicate probability materialization
Unimplemented; no speed or accuracy result claimed.
The complete exact pointwise decoder now reproduces every recorded original
output and trace on its development qualification, but saves only about9%
at128 positions and2.5% at40. It still misses the overall reconstruction goal.

Inspect the existing forward/backward probability lifetime and the optimizer's
second softmax of unchanged logits. Determine whether the already computed
probability can be reused without changing dtype, shape, stride or operation
order. Do not substitute exp(log_softmax) for softmax; their bits may differ.
Do not derive log probabilities by taking log of rounded probabilities.

First independently qualify exact reused probability arrays and backward
gradients on public cold/intermediate/nearly converged full-vocabulary states,
including original0/16/64/128-step fixtures. Keep native reductions, constants
and all128256 coordinates. Preserve any capture-lifetime or output difference.
Use an owned capture stream and explicit input lifetime; no cross-stream
autograd retention. Numerical equality must be measured, not presumed.

Only advance after a written implementation/preflight and largest native
geometry qualification. Then repeat complete decoder output/trace checks and
paired timing with source bindings and freeze-before-scoring. This execution
direction alone cannot repair the unresolved reconstruction errors and cannot
justify a baseline replacement claim or canonical score inheritance.

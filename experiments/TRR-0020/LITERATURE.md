# Primary literature checked
[Behrmann et al., Invertible Residual Networks (ICML 2019)](https://arxiv.org/abs/1811.00995) studies residual networks whose training enforces invertibility. It does not establish that the unmodified public Llama prefix is invertible or that a fixed-point solver converges here. Our full-vocabulary residual correction is an explicitly approximate experimental method with no such guarantee.

The previous layerwise L-BFGS and joint inverse results in TRR-0017 remain the direct evidence for this particular prefix. The current hypothesis differs by staying on vocabulary entries and correcting an analytically factored full-vocabulary score instead of optimizing an unconstrained continuous input.

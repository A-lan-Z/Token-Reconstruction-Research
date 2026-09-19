# Development14: test the optimizer floor before changing the search representation

This prospective study returns to the full-vocabulary Gini003 rule. Query-space optimization lost accuracy. The canonical Gini128 rule was also too inaccurate, so reducing its iteration count requires a convergence improvement.

Hypothesis: Adam's fixed denominator epsilon suppresses updates to low-probability vocabulary coordinates whose gradients are small. An average-over-positions loss also scales those gradients down as input length grows. This could delay recovery of initially unlikely tokens even though the whole vocabulary is present. This is a mathematical hypothesis; it has not been established as the empirical cause of the current errors.

Keep all128256 logits at every unknown position, the supplied frozen prefix, scale80 metric initialization, Gini.003 after32, adaptive position rate, decay.98, moments(.9,.995), reset32, no bias correction and128updates. Change only the base rate (.15/.3/.6) and denominator floor (1e-12,1e-16,1e-20,or1e-12 divided by unknown-position count). These12fixed exploratory configurations create96cells on the same8retrospective inputs. Preserve all ordinary snapshots and objective-selected outputs. No candidate proposals, vocabulary pruning, new fitted model or separate verifier.

The length-scaled rule is motivated by the Adam identity: multiplying every gradient by c with zero initial moments is equivalent to dividing epsilon by c. A CPUfloat64 check verifies this and illustrates floor suppression on synthetic small gradients. It is not a measurement of our model's gradients.

Before GPU execution: qualify the new Triton update against a direct FP64 formula across gradients covering the tested magnitudes. The fixed1e-12/rate.6 control must reproduce the original Gini003 outputs and traces on the public largest fixture and the8development inputs. Use deterministic flags, native geometry, and one fresh process per configuration to avoid the process-local memory accumulation observed in development13. Save all qualification outputs and traces before gates. Full96cell freeze before current labels. Reuse no previously scored answers for decisions.

Geometry128positions by128256logits, hidden2048; existing Gini implementation qualified at3.807GiB on the development geometry. Keep6GiBreservedcap,minimum2GiBfree, admission9000MiBfree,hostavailable8GiB,RSS10GiB,temp80C. Largest128case before each configuration, check every real cell as well. Estimate under5min,timeout1200. Do not launch concurrently with development13_r1. This is not registered as an active canonical method.


Numerical preflight correction before any reconstruction run: the111-coordinate test passed at1e-12 and1e-16, but1e-20 showed an absolute step error of5.15147 against FP64 when a1e-18 gradient's variance entered the subnormal range. Preserve dev14_kernel_reference.json as a failed qualification for the unscaled implementation.

For every reduced-epsilon mode, multiply the gradient by1e6 inside the optimizer, accumulate the correspondingly scaled first and second moments, and multiply epsilon by1e6 as well. This is algebraically the same Adam ratio but avoids underflow in the variance calculation. Keep the1e-12 control unscaled to retain exact original arithmetic. Prefix gradients and loss calculations are unchanged. This is explicitly recorded FP32 arithmetic, to be requalified before the matrix. Do not silently use the failed unscaled1e-20 update.

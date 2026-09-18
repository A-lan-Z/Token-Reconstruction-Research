# Development ledger

## Forward-response cache — measured negative
At code 5f9cdd7b52926705e717f35acf5a68ccc1f42e40, the 3x16-token opened development panel gave 32/45 Euclidean, 33/45 cosine, and 31/45 context-transport reconstruction, versus 45/45 A1+A2. No challenger reconstructed a complete clip. Cache construction 2.556 s, 1.051 GB. Current-context corrections did not repair enough omitted candidates. Preserve all 12 frozen predictions and the separate score; this is not a promising result.

## Next structural hypothesis: suppress directions that residual branches write strongly
For each prefix attention output matrix and MLP down matrix W, compute W W^T divided by its trace. Sum these weight-derived matrices. Its eigensystem identifies directions the prefix can write strongly; a Mahalanobis token metric downweights those directions and emphasizes residual-stream coordinates least changed by contextual writes. Query and token embeddings are transformed identically. This is a deterministic function of the same prefix weights, not an auxiliary-data-fitted inverse or calibration model. It has no independently adjustable learned parameters and must be invalidated on weight changes.

Three fixed development probes: total output-Gram inverse-square-root metric; a diagonal-only version; and MLP-only output-Gram metric. Full-vocabulary cosine proposals, K64, then unchanged current-prefix forward verification. These are distinct algebraic hypotheses, not an exhaustive threshold search. Native output/Gram matrices are <=2048x2048 (~16 MiB FP32); eight live 2048x8192 FP32 matrices are not retained. One transformed vocabulary table ~1.05 GB. Eigensolver plus transform is expected within 30 s and peak below the already qualified 6 GiB cap; the watchdog is 180 s. No backward graphs.

This may fail because full-rank writes overlap the embedding signal, because nonlinear residuals rotate rather than merely add nuisance components, or because Gram amplitudes do not reflect actual execution. Any improvement is a proposal mechanism signal, not evidence of exact inversion.

# Model-native nuisance geometry

Centering and Euclidean geometry do not close the natural-text gap; preserve all288 results. Equal-trace output-Gram whitening was an approximation to contextual nuisance, not a privileged metric.

Test three structurally defined alternatives with both embedding and intrinsic-MLP dictionaries:
1. Attention-only equal-trace output Gram: MLP encoding is signal in the intrinsic table, so only contextual write directions are suppressed.
2. Unnormalized total write Gram: preserves the prefix's native relative weight scales.
3. Exact full-vocabulary response-difference covariance: R_v=P([BOS,v])_1-D_v, compute mean and covariance of these exhaustive same-prefix responses, subtract that mean from query and whiten by covariance. D is embedding or MLP-only. This is an algebraic cache of prefix forward responses, not regression to text labels, an optimized decoder, or an independent checkpoint. Zero external examples and zero optimizer steps; it must be discarded on any prefix update. This last variant has a larger rebuild cost that must be included if selected.

Top256 proposal-only outputs on all48 opened R1 observations for six arms; freeze all288 before scoring. No test-truth statistics enter candidate generation. Fixed spectral floor float32 epsilon times max eigenvalue; no ridge sweep. Preflight prefix+BF16 intrinsic+.FP32 table+FP32 response ~3.8GB and small Gram/chunk temporaries, within6GB guard. Dictionary256 numerical contract retained; covariance chunks4096 are an explicitly declared approximate numerical reduction, not a microbatch workaround. Watchdog120s and permetric guard.

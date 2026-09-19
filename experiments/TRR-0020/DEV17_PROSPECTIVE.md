# Prospective alternative: cache a low-rank local quadratic over the full vocabulary

Status: mathematical CPU prototype only, no GPU harness or reconstruction run. The active canonical_refine experiment and the already-prepared development16 sensitivity study take priority. No new active method is registered.

Development16 compresses normalized-prefix sensitivity into one isotropic scalar per position. A stronger local calculation can retain directional information from the same8 vector-Jacobian probes without requiring more per-update vocabulary products. This is a hypothesis, not evidence of reconstruction improvement.

For each position let G contain the normalized-prefix probe gradients and R be the fixed raw/white metric transform. Define lambda as the mean squared norm of G R^{-T}, divided by dimension. Form a positive semidefinite curvature B=(1-alpha)G^T G/S + alpha*lambda*R R^T. Positive alpha supplies curvature in directions not covered by the small probe set. It remains an approximation, not a global Hessian bound.

For each vocabulary embedding e cache q(e)=e^T B e once at the warm starting point. Streaming over probes avoids materializing a positions-by-probes-by-vocabulary tensor. At each subsequent update calculate Bz from the small stored probe matrix, then score EVERY vocabulary token as (scale*Bz-gradient).e - .5*scale*q(e). This is the direct full-vocabulary quadratic minimizer. The cached surface adds no shortlist, and no token alternatives are passed to a separate verifier.

The cache needs127x128256FP32 values (62.14MiB), and8probe vectors need7.94MiB. Building it uses8vocabulary matrix products once per input; each update still uses one product. A proposed FP32 scoring implementation would retain another1002MiB embedding copy and needs a new largest-cell resource qualification under the6GiB cap. Full preparation and per-input cache costs must be timed. Numeric geometry must be explicitly fixed and qualified, since different batching/precision can alter token choices.

The CPU test compares streamed cached scores against explicit dense positive-definite quadratic matrices for alpha.1/.5/1 and scales.1/1. It checks identical argmax over every entry and a linear identity example including the final vocabulary token. These checks prove the algebra on the tiny test, not the quality or runtime of the Llama-prefix approximation.

If development16 establishes useful sensitivity behavior, consider a separately frozen grid of warm length, metric, shrinkage and multiplier. Declare that grid before any reconstruction, preserve failed attempts, freeze every cell before labels, and require both canonical benchmarks before a replacement claim. No grid has yet been selected for this alternative.

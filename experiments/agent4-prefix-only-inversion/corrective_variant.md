# One justified corrective variant, before final source selection

The initial bounded SGD procedure failed the one-token public CPU BF16 search
fixture after128 checks (12.07s). Continuous MSE moved from0.01136 to0.00889;
minimum verified MSE was0.01046. Input gradients were finite and continuous /
discrete paths exact. This diagnoses poor bounded optimization progress, not
an implementation disconnection or evidence of information loss.

The single corrective variant uses Adam lr0.01 with its default betas/eps on
the same one-token continuous variable, retains128 checks/gradients, and removes
periodic snapping and decayed SGD steps. Same initialization seed, raw Euclidean
projection and exclusion of tried tokens. No restarts, no inverse initialization,
no truth-based early exit. The variant must be frozen before unused evaluation
source selection. This is a bounded optimization correction, not a sweep.

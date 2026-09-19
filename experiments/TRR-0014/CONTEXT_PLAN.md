# Parallel context stripping with a prefix-native intrinsic dictionary

MLP-only dictionary K64 includes all312 stress tokens in both target conditions, but its union with embedding-metric K64 still misses23/2032 matched and19/2032 shifted natural tokens. Not yet a replacement.

New candidate-generation mechanism: compute the contextual contribution of the supplied prefix on a tentative full sequence, subtract it from the observation, and re-query its deterministic MLP-only token table. If D_v is that intrinsic response and z is the tentative sequence, q_i=H_i-P(z)_i+D_(z_i). All operations use the same prefix and permitted H. No training, auxiliary predictor, backward pass, or current truth. Iterations across the full record make candidate generation a small fixed number of matrix operations rather than a per-token optimization.

Preregistered development probe: initialize from embedding-metric top1 and intrinsic-metric top1 separately. Four forward-only context-stripping rounds per initialization, collect16 candidates each round. Always retain initial32 candidates from each dictionary. Branch-only128 and union192 candidate-recall diagnostics. BOS fixed128000. Use the unchanged metric and direct lookup; no correctness-based stopping. All48 opened R1 observations and all144 files frozen before scoring.

Preflight: prefix~1GB, twoFP32 normalized tables~2.1GB, intrinsicBF16 table~.53GB, small sequence intermediates, expected<4.5GB versus6GB guard. Qualify max128-position fixture before matrix. Full sequence forward is an explicitly approximate proposal operation, not a semantic replacement for native A2. Final A2 would retain its native numerical geometry and candidate selector. Watchdog120s, no gradients. Later timing must include dictionary rebuild if weights change.

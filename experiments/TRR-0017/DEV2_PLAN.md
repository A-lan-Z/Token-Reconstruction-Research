# Development follow-up 2 (frozen before execution)

The full dev1 matrix is frozen and scored. Layerwise L-BFGS is inaccurate (matched prose cosine84/254 at96 steps), while joint96 L-BFGS reaches177/254 prose and78/78 identifiers but costs1-2s. More optimizer iterations alone are not adequate. These unsuccessful probes remain preserved.

Test one evolving whole-sequence estimate initialized by the prefix-weight metric full-vocabulary argmax. This initialization adds a deterministic cache derived from the same prefix; no fitted A1 and no candidate shortlist. Evaluate native BF16 forward/gradient rather than the explicit FP32 approximate inversion, and rescale the output loss with the same prefix-derived write metric to reduce conditioning problems. This is a new numerical method, not a native-equivalent A2 optimization.

Five fixed variants: Adam64 lr0.003 raw residual; Adam64 lr0.003 whitened residual; same with whole-sequence cosine projection every8 steps and optimizer reset; L-BFGS32 and96 whitened residual. L-BFGS history8, strong-Wolfe line search, max_eval2xiterations, tolerance_grad1e-8/tolerance_change1e-10. Choose lowest observed continuous forward loss iterate, final full-vocabulary cosine projection. Known BOS only. No candidate verification, truth-dependent stopping or routing.

Same8 development records and128-position qualification fixture as dev1, all40 outputs frozen before scoring. Metric1.05GB+FP32 embeddings1.05GB+normalized embeddings1.05GB+BF16 prefix~1GB; largest projection128x128256~66MB, gradient workspace<0.5GB. Expected peak<6GiB, guard unchanged. Dev1 largest qualified reserve<6GiB. Expected<2min under watchdog900s.

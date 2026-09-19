# Whole-observation inverse residual proposal probe

The generated-response dictionary in weight geometry recovered310/312 stress tokens in both conditions but only1420/2032 and1433/2032 natural tokens. Its union with embedding-metric K64 has only2009/2032 and2011/2032 natural proposal recall; preserve this negative as a general replacement.

Next distinct computational route: reverse the prefix's eight additive sublayers using only forward computation, over all observed positions together. Reverse each MLP then attention sublayer using x <- y - f(x), with either six damped half-steps or ten Anderson acceleration steps (history4, relative Gram diagonal1e-4). Retain the iterate having smallest allowed local equation residual per position, and restore exact supplied-prefix BOS states. No backward graph, candidate-token gradients, auxiliary fitted weights or target calls. Test raw embedding cosine and weight-metric cosine K64 after reversal.

All48 opened R1 observations; all192 proposal outputs frozen before scorer. This tests recall only: emitted token is proposal rank1, no A2 quality or final runtime claim. Timing is rough, includes inverse and proposal, and metric timing includes preceding raw projection. If promising, a final policy must be timed correctly and freshly confirmed.

Preflight: largest128x2048 observations, four history vectors perposition (~4MiB), small4x4 batched solves,128x8192 MLP activation(~2MiB BF16), prefix+twoFP32 vocab tables~3.2GB. Within earlier qualified4.4GB reserve and6GB guard. First run both inverse geometries on a full128positions synthetic fixture before releasing matrix. Watchdog180s; same native BF16 forward sublayers, no equivalence claim to exact inversion.

# Development5: graph execution and magnitude-aware full-vocabulary inverse
Dev4 is complete/scored and does not improve dev3 prose quality. Keep full-sequence gradients. Dev3 reset80/lr0.6/decay0.98 reaches252/254matched prose and78/78identifiers; quality and speed still unqualified.

Implement CUDA graphs for the SAME full-vocabulary softmax, expected embedding, prefix forward/backward and reset-moment update. Fixed maximum logit storage127x128256; each capture slices to its actual native sequence length. No padding of the prefix forward pass. Separate private graph pools; cache at most2lengths and count any recapture in runtime. Prefix weights frozen. Starting logits remain80xfull-vocabulary weight-metric scores built/calculated in FP32 with TF32 disabled. No shortlist or hard-token A2 reranker.

One FP32 graph cosine control (LR0.6) must reproduce the original ResetSoftVocabulary saved output snapshots/best sequence and synthetic loss trace; compare length128,40,83,128 with a new public synthetic fixture (seed200025), including graph eviction/reuse. If it differs, preserve/exclude the exact-execution claim and diagnose before accepting it.

TF32 is a separately labeled numerical variant used ONLY during captured optimization, never during metric/initial-score construction. Primary API references: https://docs.pytorch.org/docs/2.10/notes/cuda.html and https://docs.pytorch.org/docs/2.10/generated/torch.mm.html . No assumption of numerical neutrality. Register any eventual winner with its exact numerical rule.

Fixed10development configurations, all256steps,init80,betas(.9,.995),epsilon1e-12,decay.98,reset32:
- cosine FP32 LR.6
- cosine TF32 LR.3,.6,1.0
- relative raw MSE TF32 LR.3,.6
- cosine +0.1relative raw MSE TF32 LR.3,.6
- prefix-metric-whitened MSE TF32 LR.3,.6.
Relative MSE is mean squared error per token divided by the observed token activation's mean square (floor1e-8), then mean across unknown positions. The white transform matches the prior fixed weight metric normalized by Frobenius RMS. Snapshot0,16,32,64,128,256 and best soft objective. Magnitude losses address fractional mixtures that can have the right direction without the right vector.

80cells on the same8opened development inputs, plus public max128qualification before eachfamily; freeze allbefore scorer labels. No canonical claims yet. Compare FP32graph cosine outputs to dev3sameconfig on every shared input, preserve any differences.
Resource estimate:prefix~1GiB,metric1.05GB,FP32embedding1.05GB, four max-size logit/state/gradient buffers~0.27GB, graph pools for max2lengths<=1.5GB. Require<=6GiBreserved and>=2GiBfree; watchdog unchanged. If cap fails stop and reassess. Geometry is never shrunk to evade guard. Runtime expected<10min,timeout1200s. Record setup/capture/cache eviction time, inference,model evaluations,peak memory,and failed attempts.

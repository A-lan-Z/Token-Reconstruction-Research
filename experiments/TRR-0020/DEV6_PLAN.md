# Development6: bandwidth-efficient full-vocabulary optimization
Dev5 complete/scored: no effective replacement yet. FP32graph reproduced original token outputs and all synthetic loss values exactly; TF32 is a numerical variant. On identical development inputs, historical A1+A2 recovers every token (verified output/hash bridge). Strongest new methods still miss prose tokens. GPU operator profile:~8.18ms/update; two full-vocabulary FP32 matrices~2.60ms, significant additional memory traffic in separate optimizer updates.

New explicitly approximate numerical method: keep FP32 logits/probabilities/moments, but multiply BF16-rounded probabilities by the prefix's native BF16 embedding table, accumulating/outputting FP32. Backpropagate the mixture with BF16 gradient/embedding operands and FP32 output; prefix gradients already pass a BF16 cast. Fuse moment,variance,parameter and decay updates into one Triton kernel. This is not declared byte-equivalent. All vocabulary entries participate. No top-K filter, candidate generator, hard-token verifier or fitted A1.

Cosine objective only; init80 full-vocabulary weight-metric scores remain FP32/TF32disabled. Base LR.6, betas(.9,.995),epsilon1e-12,reset32. Six fixed configurations:
- constant128 and constant256:LR.6,decay.98.
- cool128:LR.6/.3/.1 over steps0-31/32-63/64-127; decay.98/.99/.999.
- cool256:LR.6/.3/.1 over steps0-63/64-127/128-255; same decay.
- adaptive128 and adaptive256:baseLR.6,decay.98; per-position multiplier sqrt(clamp(current cosine error/.05,.01,1)).
Keep direct argmax snapshots and best full-sequence soft-objective output. Also record a declared per-position best-soft-error diagnostic: track the argmax at the lowest observed position's soft activation error. It adds no prefix evaluations and does not rerank hard token candidates; it is an alternative optimizer-iterate selection rule, requiring its own fixed registration if chosen.

Before scientific predictions, test fused optimizer vs independent torch formulas including adaptive rate and nonmultiple block sizes, and mixture forward/backward vs independent small dense calculations. Largest128publicsynthetic qualification (seed200026) plus repeatability for everyconfiguration. Preserve all failures. 48development cells on the same8opened inputs; freeze all before labels. No canonical claims.
Resources:removing the extra FP32embedding copy saves~1GiB; retained prefix~1GiB,metric~1GiB,FP32logits/state~.3GiB,at most two geometry graph pools estimated<1.5GiB. Admission>=9000MiBfree;reserved<=6GiB,free>=2GiB,temp<80;sameguard. Native forward sequence length, no new padding/batching. Matrix expected<10min,timeout1200s.

# TRR-0017: remove per-candidate prefix simulation

Request saved verbatim in coordination/requests/TRR-0017.md. TRR-0016 already exists with a different uncommitted request; it is untouched. Base commit080bcff58b0f7299d5c3504835e076dea61293fe.

## Hypotheses and development stage

The earlier sequential gradient solver repeatedly recomputed its whole growing context and verified a proposed token. The TRR-0014 reverse solver used only6 damped or10 Anderson iterations and had large layer3 attention inversion residuals. Neither tested gradient-based whole-sequence optimization with all observed positions contributing, or converged layerwise quasi-Newton inversion.

Implement two candidate-free families: (1) reverse each residual sublayer by minimizing its observed forward residual with L-BFGS, then project once against every vocabulary embedding; (2) jointly optimize every input embedding against the complete observed activation, optionally project the whole current estimate periodically. Neither fits weights, reads a fitted A1, constructs a shortlist, or runs a per-token candidate verifier. Vocabulary projection is over all128256 embeddings. Prefix evaluations act on continuous sequence estimates. No claim of equivalence to native BF16 A2; numerical choices are separately registered algorithm constants.

First qualify length128 on a public synthetic fixture. Initial exploratory constants: layer L-BFGS32/96; joint Adam128/384, lr0.01; joint L-BFGS96. Known BOS alone is clamped. Initialize layer inversion at each sublayer output; joint optimization at H scaled to embedding RMS, or at deterministic prefix-weight-metric top1 (no fitted table). Preserve all attempts. Observed residual may choose iterates; source truth may not. Before reading opened development labels, freeze all outputs for a small preregistered panel: first2 records per condition/group in the old R1 metadata (8 total). These are development variants, not registered active methods or confirmatory evidence.

Select a fixed promising method using development only, or revise with documented reasons if none qualifies. Register any final active method before execution; run both canonical setups with current A1+A2 K256 and fragment256 controls and preserve the52 inherited canonical cells with source hashes. Auxiliary evidence stays separate. Freeze full comparison outputs before retrospective scoring. Fresh confirmation required before any replacement claim.

## Resources

Live19Sep2026: RTX5080,16303MiB total,12611MiB free,43C; host29GiB available. Largest input128x2048:1MiB FP32; eight L-BFGS vector pairs<=16MiB with history8. Layer0-3 BF16 weights+embeddings about1GiB; FP32 scoring table1GiB; optional FP32 copy of prefix<2GiB. Largest MLP128x8192 is4MiB FP32; four layer gradients and attention128x128x32 stay well below1GiB. Estimated peak<6GiB reserve, >4GiB live GPU margin. No per-token full Jacobians and no vocabulary-sized transformer batch. Guard6GiB reserve,2GiB free GPU,8GiB host available,temp<80C,processRSS<10GiB. Qualify length128 before matrix. Watchdog900s development, estimated<5min. No batching optimization of existing methods is assumed neutral.

No public upload in this task; the previous automatic approval-review publication block remains unresolved in this conversation.

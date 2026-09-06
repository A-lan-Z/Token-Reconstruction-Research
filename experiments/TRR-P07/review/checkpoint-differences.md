# TRR-P07 checkpoint provenance differences

This review is metadata-only. It compares the published P06 past-only and diagonal states with the retained TRR-0006 enriched positionwise and repaired causal states. I read receipts, plans, source metadata, and hashes only; no activation, token-id, checkpoint tensor, evaluator answer, or truth payload was opened. These differences document non-equivalence of fit provenance and do not attribute any prediction difference to one factor.

## Direct comparison

| Dimension | P06 past/diagonal | Retained TRR-0006 positionwise/causal |
|---|---|---|
| Fit support | 1,200 `coverage_mix_v1` records; source IDs `69fddbc9…`; valid-mask `2decc908…`; 112,825 valid post-BOS rows after crop | 1,200 `coverage_mix_v1` records; source IDs `f3e13448…`; valid-mask `e2ee7f71…`; 124,371 valid post-BOS rows |
| Source geometry | Source artifacts `[1200,192,2048]`, cropped to `[1200,128,2048]` before schedule, mask, fitting, and selection; positions 128–191 ignored | Fit/validation receipts retain `[1200,192,2048]`; prediction replay stores/scored H128 prefixes |
| Mask | Past-only keys `0..i`, or diagonal key `i`, in H128 | Causal keys `0..i`, or diagonal key `i`, in H192 fit; prediction view is H128 |
| Initial W,b,s | Competent public affine state, file SHA `09c5b852…`; inherited `s=3.5380859375`; qkv seed 6106 or 6107 | Identity W, zero b, `s=3.0`; qkv seed 4005; no pretrained public affine loaded |
| Optimizer/budget | AdamW, lr 1e-3, wd 0, clip 1, cosine schedule; batch 8; 512 post-BOS draws/update; 3,000 steps | Same nominal optimizer and budget; shared schedule seed 4005 |
| Replicates/checkpoints | Seeds 6106/6107. Past selected at 1700/2000; diagonal at 1700/900 | Single published states. Positionwise selected at 1600; repaired causal at 1900 |
| Selection metric | Micro full-vocabulary `validation_token_accuracy` over 2,982 H128 post-BOS rows; earliest maximum every 100 steps including step 0 | Unweighted mean of per-style post-BOS token accuracies over 3,133 H192 rows (24 Alpaca + 24 Pile); earliest maximum every 100 steps including step 0 |
| Attention score | Cosine Q/K normalization with scale 4 for both arms; one-key diagonal forward path is probability one | Retained diagonal keeps legacy dot product; retained causal is repaired `cosine_scale4` with row-wise Q/K normalization |
| Published readout precision | F32 model/logits and output normalization with shared F32 E; BF16 H input, bool masks, no autocast | Same published TRR-0006 prediction boundary: BF16 H input, F32 decoder/logits/E, bool masks, no autocast, matmul precision highest |

The nominal optimizer family and step/draw budget match, but support, crop, initialization, seed/schedule, selected checkpoint, selection metric, and (for the retained diagonal) declared score rule differ. Equal record counts therefore do not establish equal fit data.

## State bindings

- P06 past seed 6106: `experiments/TRR-P06/runtime/main-r1/seed-6106/p06_past_only/selected.safetensors`, SHA `a1406fcb…`.
- P06 past seed 6107: `experiments/TRR-P06/runtime/main-r1/seed-6107/p06_past_only/selected.safetensors`, SHA `6ad3b8e9…`.
- P06 diagonal seed 6106: `experiments/TRR-P06/runtime/main-r1/seed-6106/p06_positionwise_diagonal/selected.safetensors`, SHA `34b9b0cf…`.
- P06 diagonal seed 6107: `experiments/TRR-P06/runtime/main-r1/seed-6107/p06_positionwise_diagonal/selected.safetensors`, SHA `ab95896c…`.
- Retained TRR-0006 positionwise: `experiments/TRR-0005/joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors`, SHA `696eb9fc…`.
- Retained TRR-0006 causal: `experiments/TRR-0005/joint_fit_qknorm_v1/enriched/affine_causal_h_attention128/selected.safetensors`, SHA `ee910b14…`.

The full hashes, source commits, per-seed initial-state hashes, schedule hashes, and receipt paths are in the companion JSON.

## Truth boundary and limits

The already-opened P06 truth metadata is `/tmp/trr-p06-evaluator-truth-v1/truth.manifest.json` (SHA `21c07d64…`). The published TRR-0006 score command is `experiments/TRR-0006/score_command.json` (SHA `5e28a3bc…`) and binds the metadata-only path `/tmp/trr0006/private/truth_binding.json` (SHA `c2f96e87…`). The underlying truth payload was not opened for this review.

No tensor equality, source-record intersection, training-time autocast equivalence, or causal attribution is claimed. The old qk amendment explicitly retained the diagonal state without cosine repair; the repaired causal state is the separate cosine-normalized fit selected at step 1900.

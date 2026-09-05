# TRR-0004 initial public decoder diagnostic

This is a post hoc development diagnostic of the three selected TRR-0003 checkpoints. It uses the disjoint public validation slice (24 records, 936 post-BOS positions) and public fitting labels for coverage bins. It does not open current panel/evaluator truth, load target weights, call a public prefix, or use A2/candidate fallback. It is not independent confirmation and it does not change the deployed prediction artifacts.

The historical-inputlens bridge control also passed its bit-exact implementation check on the same public validation observations and normalized embedding table: projection and logits were exact-equal, and the top-1/top-512 ranks matched for all 960 checked rows (0 mismatches). This verifies the W.T orientation, float32 projection, normalization, embedding-dtype cast, and exp(s) convention against the retained reference; it is an implementation control, not target-transfer evidence. The result is recorded in `experiments/TRR-0004/bridge/public_validation_equivalence_v1.json` (the launcher receipt is `experiments/TRR-0004/bridge/bridge_run_v1.receipt.json`; both were run at commit `328f3a3559289338b7e38fc64465e1808d466e29`).

The fitted vocabulary bias is a partial contributor to the CE gap. Zeroing the fitted bias after fitting improved the tied affine checkpoint from 587/936 (62.71%) to 602/936 (64.32%), and the MLP checkpoint from 506/936 (54.06%) to 540/936 (57.69%). The corresponding seen/unseen counts were:

| checkpoint | control | all | seen (510) | unseen (426) | exact records |
| --- | --- | ---: | ---: | ---: | ---: |
| angular control | native / no bias (no bias parameter) | 590 | 435 | 155 | 0/24 |
| tied affine CE | fitted bias | 587 | 441 | 146 | 0/24 |
| tied affine CE | fitted bias zeroed post hoc | 602 | 442 | 160 | 0/24 |
| residual MLP CE | fitted bias | 506 | 439 | 67 | 0/24 |
| residual MLP CE | fitted bias zeroed post hoc | 540 | 434 | 106 | 0/24 |

The MLP gain is concentrated in unseen validation tokens (+39) while losing five seen tokens; tied affine gains 14 unseen and one seen token. The fitted bias distributions had all-vocabulary means of -0.1531 (tied affine) and -0.3629 (MLP), with standard deviations 0.0282 and 0.0662. These are descriptive associations, not a causal explanation of how a no-bias model would fit.

The scale and normalization controls are inference-only ablations. With the fitted bias retained, changing scale from 16 to 1 reduced tied affine accuracy to 429/936 and MLP accuracy to 416/936, with zero unseen hits in both cases; this exposes the interaction between the learned bias magnitude and scale. With bias zeroed, positive scale change preserved predictions exactly for all 936 rows. Disabling output normalization after also zeroing bias preserved CE token predictions (602 and 540 respectively); it is therefore not a demonstrated source of this checkpoint gap. Angular token predictions were unchanged by all controls, although one unnormalized rank aggregate moved slightly.

The guarded run used commit `328f3a3559289338b7e38fc64465e1808d466e29`, Torch `2.10.0+cu128`, CUDA 12.8, an RTX 5080, and batch size 128. The launcher exited 0 at 2026-09-05T10:35:26Z–10:35:29Z; the diagnostic receipt's measured wall time was 1.3836 s. Public input/record loading took 0.4407 s, one-time embedding transfer 0.1904 s, and method projection totals were 0.3249 s (angular), 0.1740 s (tied), and 0.1898 s (MLP). Peak GPU allocation was 1,300,207,616 B and peak reservation 1,493,172,224 B; the shared embedding boundary held 1,050,673,152 B. No compute process remained afterward.

Receipts:

- Diagnostic JSON: `experiments/TRR-0004/track_b/bias_diagnostic_run_v1/public_decoder_bias_diagnostic_v1.json` (SHA-256 `a20f88e301a6f8e5c7bd4bab8f3c9f3da3fd81b4d696e661327327e0ea23d15d`)
- Launcher receipt: `experiments/TRR-0004/track_b/bias_diagnostic_run_v1/run_receipt.json` (SHA-256 `e5b7e82cd63edcd8d314d86db0f9f0e2a0dc7be394eca19463f29b94a6cceaaa`)
- Standard output/stderr: `stdout.log` and `stderr.log` in the same run directory; stderr was empty.

The result leaves a real but incomplete hypothesis: fitted vocabulary calibration contributes to the CE standalone gap, especially for unseen tokens, but the diagnostic does not distinguish bias capacity from other fitted-state effects. The next stage is controlled nested fits, then context if the competent affine base is established: keep that base explicit and test added causal H-only context against a parameter-matched positionwise nonlinear path.

# Development20: focus the unchanged full-vocabulary optimizer on unresolved errors

Development19 completed32cells with128readout variants. None improves group accuracy; cosine readouts sometimes switch between wrong tokens and retain the same scores, while Euclidean readouts lose two further warm64shifted prose tokens. Thus simply changing the readout is insufficient. Frozen-state inspection shows every warm64error has soft cosine error>.05; the four remaining warm128shifted errors have errors.176to.664and mixed confidences. This motivates residual weighting, not an assumption that all errors are overconfident.

## Fixed algorithm
Keep the complete Gini003 architecture, all128256vocabulary entries, initialization, BF16mixture, FP32fused optimizer, learning-rate adaptation, moment resets, decay and Gini coefficient. Through steps0..31 use the original loss. From step32 use mean(max(error,0)^p)/(p*.05^(p-1)) plus the unchanged.003mean-Gini term. Here error is each position's observed cosine error from the same existing prefix pass. Powers are1(control),2or4. Power1 takes the exact original code path. At error.05the new error-gradient coefficient equals the original; smaller errors get less weight and larger errors more. This changes the objective, not just its execution, and is not a claim of unbiased gradients or guaranteed convergence. Numerical negative cosine errors are clamped only in the powered terms.

No model parameters are updated. There is no proposer, top-k, verifier, extra A2call, source-truth signal, candidate ban or vocabulary pruning. Every token remains available at every optimized position. Emit the unchanged final-logit argmax and preserve the existing objective/error selections as exploratory outputs.

## Matrix and controls
Six fixed configurations:64or128updates crossed with powers1,2,4; all eight established development inputs per configuration,48cells total. Run the largest steps128/power4cell first, then the remaining isolated workers. Every configuration repeats the fixed public fixture seed200041 three times at native128and40lengths. All tokens and all four traces must repeat exactly.

Both power1controls must reproduce their full archived soft trajectories on all16real cells and both public geometries. New powers must reproduce the untouched starting segment: snapshots0/16/32, loss traces through step31, and observed-error/confidence/Gini traces through step32before the first weighted update. Public anchors come from the corresponding dev19warm phase; real anchors are the unchanged dev7Gini003 outputs. CPUdoubleprecision tests verify the normalized-output loss chain rule against explicit per-position gradient weights for each power. Save all public and reconstruction outputs before gates. A gate failure is preserved and excluded, not relaxed after observing labels.

Freeze all48cells before opening labels. The scorer must validate sources, assets, geometry, complete grid, anchors, outputs and three negative gate tests. This is retrospective development; no active canonical method is added. Any proposed replacement must complete both canonical setups and the entire active matrix with fresh timing controls.

## Cost and resources
The number of prefix calls stayssteps+1forwards andstepsbackwards, as in the original Gini method. Additional work is scalar elementwise arithmetic on at most127errors. Preparation and replay capture remain separately timed and total decoding includes synchronization and output copies.

Largest native geometry is128tokens with127x128256FP32logits and the same optimizer state. There is no additional FP32vocabulary copy or readout table. Based on earlier owned-stream Gini/refinement measurements, estimate<=4.2GiB peak, cap6GiB, GPU free>=2GiB, host available>=8GiB,RSS<=10GiB,temp<80C. Expected<=240seconds; guard timeout600seconds. Live preflight is in dev20_preflight.json. Do not overlap other GPU work or change batching as an unqualified resource workaround.

Alternative disposition: matrix-free Newton-Krylov is still available but needs a new derivative implementation and extra model-vector operations. Reheating logits is also possible, but the readout diagnostics do not support overconfidence as the sole failure mode. Residual weighting is tested first because it addresses both high- and low-confidence unresolved states without adding prefix evaluations.

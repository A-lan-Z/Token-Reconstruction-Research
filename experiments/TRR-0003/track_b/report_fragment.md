# TRR-0003 Track B result fragment

## What the public-only pilot learned

Direct token supervision fit the public observations, but held-out public transfer was the limiting step. The eight-record, 1,500-step overfit diagnostic reached 100% token accuracy for the angular control, tied affine CE decoder, and residual MLP-256 CE decoder, so the basic optimizers and parameterizations can fit a small sample. On the disjoint 24-record validation slice, the best public-only checkpoints were angular step 150 at 63.0342%, tied affine step 175 at 62.7137%, and MLP step 375 at 54.0598%. At the predeclared 1,800-step endpoint, train/validation accuracy was 99.0785%/52.2436% for angular, 100%/46.4744% for tied affine, and 99.9199%/48.1838% for MLP. The learning curves therefore separate fitting from generalization and justify selecting the early checkpoints by public validation; they do not establish a canonical benchmark result.

The standalone decoder test is direct token prediction with no A2 fallback or candidate-by-candidate public-prefix simulation. Both CE decoders use full-vocabulary cross-entropy and a normalized fixed public embedding table for the final projection. The tied affine arm is consequently a compact, structurally constrained decoder rather than an unrestricted 2048-by-128,256 classifier. The MLP adds a 256-wide nonlinear residual and did not improve held-out transfer in this pilot. The angular inverse is retained as the existing fitted control.

A public-label transfer diagnostic found 510/936 validation positions (54.4872%) whose token type appeared in the fit labels. The fit and validation vocabularies overlap on 226/594 unique validation types (38.0471%). This is a plausible contributor to the ceiling, rather than an established causal explanation: angular accuracy was 85.2941% on seen positions and 36.3850% on unseen positions; tied affine was 86.4706% and 34.2723%; MLP was 86.0784% and 15.7277%. The diagnostic used only public fit and validation labels and did not inspect shared-panel truth.

## Learning curves, selected states, and runtime

The standard curve figure is generated from `outputs/TRR-0003/track_b/extended_fit_1800_v1/fit_evidence.json` by `experiments/TRR-0003/track_b/plot_learning_curves.py`:

- `outputs/TRR-0003/track_b/figures_v1/track_b_learning_curves.png` (SHA-256 `d797c94a433510cbab0e5ae206a3017217e8ac71e64c95a2e2c4112b35ac659c`)
- `outputs/TRR-0003/track_b/figures_v1/track_b_learning_curves.svg` (SHA-256 `d7a087e3c75b8ccd58b8a7ab9cffddd7aa5886566a5b34d9166177212e7d3deb`)

| method | selected step | validation accuracy | fit accuracy at selected step | tiny overfit accuracy | selected state bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| angular inverse control | 150 | 0.630342 | 0.974960 | 1.000000 | 16,785,688 |
| tied affine CE | 175 | 0.627137 | 0.997396 | 1.000000 | 17,299,160 |
| residual MLP-256 CE | 375 | 0.540598 | 0.955329 | 1.000000 | 4,717,416 |

The fixed normalized public embedding table is 1,050,673,488 bytes in FP32 and is required by all three deployed prediction paths. It dominates the retained runtime footprint; the MLP decoder state is small but did not transfer as well. Selected state SHA-256 values are `e595f3f49365d8b2362972e3d43834856a0e01a722e1bd563e7de52e11b683dc` (angular), `0dad94bc31c39654506ac5ba9b75dc5cd64ef98935dc8372032bb03237509c28` (tied), and `203af5ddab65eee11b1faa8fea5976154704a8fe56bb5480b1895c92564c0c86` (MLP).

The selected-state panel adapter produced all four cells for all three methods before any private truth was opened. It used direct argmax, zero candidate simulations, zero public-prefix calls, and no target weights or source tokens. Warm repeats were exactly equal on every cell. Aggregate first/warm inference times and peak device memory were:

| method | first inference, 4 cells (s) | warm repeat, 4 cells (s) | max CUDA allocated (B) | max CUDA reserved (B) |
| --- | ---: | ---: | ---: | ---: |
| angular inverse control | 0.236841 | 0.071564 | 1,390,943,232 | 1,933,574,144 |
| tied affine CE | 0.082533 | 0.078101 | 1,908,001,792 | 1,933,574,144 |
| residual MLP-256 CE | 0.170537 | 0.081327 | 1,908,001,792 | 1,933,574,144 |

Panel startup was separated from steady-state inference: embedding load/validation took 0.364146 s, byte hashing 0.945119 s, and one-time device transfer 0.388938 s. The guarded panel process took 30.063409 s including model/state loading and serialization. Evidence is `outputs/TRR-0003/track_b/panel_selected_v1/prediction_evidence.json` (SHA-256 `df0fafe090a2c55d31a4d8fbd937dbed1f6deed477d9e60b395ad55ad36033c2`); its status is `PUBLIC_PREDICTIONS_COMPLETE_BEFORE_FREEZE`, with `truth_opened: false`.

## Preparation and training cost

The public fit used 128 records and 4,992 post-BOS positions. The disjoint validation slice used 24 records and 936 positions. The following wall times are from the raw guarded evidence; the 600-step and 1,800-step rows include each run's separate 1,500-step tiny overfit diagnostic.

| phase | wall time (s) | peak CUDA allocated / reserved (B) | role |
| --- | ---: | ---: | --- |
| shared public preparation | 4.750630 | 2,609,591,296 / 2,625,634,304 | extract fit observations and normalized embedding table |
| 600-step exploratory fits + tiny diagnostics | 108.444763 | 3,955,859,456 / 4,007,657,472 | initial learning curves and subset-fit check |
| 1,800-step extension + tiny diagnostics | 200.670264 | 3,955,859,456 / 4,007,657,472 | predeclared longer curve and checkpoint selection |
| selected checkpoint replay | 26.542253 | 3,955,859,456 / 4,007,657,472 | public-only selected-state materialization |
| seen/unseen transfer diagnostic | 2.965303 | recorded per method in diagnostic | public-label analysis only |
| selected panel prediction | 30.063409 | recorded per cell above | direct deployment-path timing; truth remained closed |

The 4.750630-second preparation and 26.542253-second replay are the relevant public setup costs for selected decoders. The 600-step fits, 1,800-step extension, and tiny diagnostics are pilot exploration costs retained for learning-curve and capacity evidence. These rows are reported separately rather than summed: they have different scopes, the fit rows include repeated tiny diagnostics, and qualification, preserved failed attempts, guard startup, and serialization are recorded in their own evidence. Use the individual guarded artifacts for resource accounting. The replay selected angular150, tied175, and MLP375 using the fixed earliest-tie rule from public validation. Replay evidence is `outputs/TRR-0003/track_b/selected_checkpoints_v1/selected_replay_evidence.json` (SHA-256 `b448a8d13e3cf14b5e74305c79df696e59a889d4ff7890f608acb16ed4c778d0`), and its guard passed with source commit `770f3d833df42f5ac98f423c3e459dbdb74a9e41`.

## Evidence and scope

The public fit, validation labels, selected replay, and seen/unseen diagnostic are disjoint from the shared 16-record panel. Validation labels were legitimate public auxiliary data; shared-panel truth remains evaluator-private. The transfer diagnostic is `experiments/TRR-0003/track_b/token_transfer_diagnostic_v1.json` (SHA-256 `052e66486988a6b8b990bf8d3f232a14388e1a950ec2d470a81c2e944e0ebaab`) and its guard passed. The panel prediction guard is `experiments/TRR-0003/track_b/predict_cells_guard_v1.json` (SHA-256 `76dd39fb56eaa016f71729e63efb276bb73a32de7df1e6f804e43cd35783186`); this is a prediction-complete, pre-truth artifact, not a scored accuracy result. The canonical dual-benchmark matrix for new methods was not run, so these findings remain exploratory.

The shared comparator bundle produced alongside this fragment is `outputs/TRR-0003/footing/comparator_matrix_v2/`, with guard `experiments/TRR-0003/footing/comparator_matrix_guard_v2.json` and run evidence SHA-256 `07c00066e760512278af08827be4d665100750ec756561d6d6b4f24d151c67fb`; it was executed at commit `b169134f7b97eb4447ad92369fb08909aeea25b7` and is handed to the shared footing scorer.

These results support a next experiment that expands public-data diversity and tests whether a decoder trained on broader token types improves matched and shifted target transfer while retaining the no-A2 runtime. A change of direction would be warranted if independent held-out and shifted-target panels remain near the current 50–63% ceiling after enough diverse public supervision, or if the one-gigabyte fixed embedding table makes the compact decoder’s runtime footprint unacceptable. No method should be promoted as a replacement from this pilot alone.

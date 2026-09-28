# TRR-P12 geometry-only review

Status: `GEOMETRY_COMPLETE_NO_EVALUATOR_TRUTH`. The frozen B1 decoder was scored against the public embedding table for all 128 records, 127 post-BOS positions, and stages 0/64/128/256. The actual boundary rows are retrospective diagnostics; the early forecast was frozen after stages 0/64 and before later-stage feature reads.

| Domain | 0→64 changes/crossings | 0→128 changes/crossings | 0→256 changes/crossings | ties |
|---|---:|---:|---:|---:|
| finance | 3/3 | 2/2 | 3/3 | 0 |
| pile | 88/88 | 84/84 | 98/98 | 0 |

| Domain | forecast positives at 128 | forecast positives at 256 | score-margin residual max |
|---|---:|---:|---:|
| finance | 14 | 994 | 0.000131607056 |
| pile | 167 | 528 | 0.000133514404 |

The public CUDA qualification passed argmax and frozen-prediction equivalence for 8/8 records and 1,016 scored rows. Its maximum CPU/CUDA score difference was 8.39233398e-05; score equality was not required and should not be reported as bitwise equality.

The forecasts are fixed 0→64 extrapolations with zero fitted thresholds. Their predictive correctness remains unknown until the post-freeze evaluator truth is scored. Boundary crossings, signed displacement, and unsigned movement must remain separate in the scientific report; this geometry result does not establish token correctness, transfer success, causality, or general fine-tuning robustness.

JSON evidence: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-P12/experiments/TRR-P12/geometry-interpretation-r1.json`.

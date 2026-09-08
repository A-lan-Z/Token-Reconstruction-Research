# TRR-0012 evaluator analysis handoff

The evaluator callback is [scripts/trr0012_transfer_analysis.py](../../scripts/trr0012_transfer_analysis.py). It reuses the existing TRR-0011 matrix gate, validates the coordinator’s frozen `expanded_fixed` 14-cell matrix and exact after-cut H/prediction null controls, then opens Agent2 truth only after those gates pass.

The frozen public inputs are recorded in `experiments/TRR-0012/transfer/analysis/analysis_handoff_v2.json`. The callback expects the Agent2 truth manifest schema `token-reconstruction.trr0012-transfer-evaluator-truth.v1`, with the public canonical panel hash and Finance/Pile ordered-ID digests repeated before either truth tensor is opened. It reports baseline-wrong, broken, improved, unchanged-correct, and unchanged-wrong inventory plus fixed higher-distortion-is-broken AUCs for raw H, relative H, margin-normalized feature shift, and actual quantized parameter relative L2. There is no threshold fitting or calibration.

Accounting is post-BOS: 127 scored tokens per record, 32 records per cell, and 4,064 scored token rows per cell. Each cell reports token-correct/token-total and exact-record-correct/32 for clean and changed predictions. Nominal parameter amplitudes remain separate from producer-measured BF16 effective relative L2; unavailable values produce `UNKNOWN`.

Validation completed without truth, source text, model loading, or GPU execution: the public matrix/null gate passed for Finance and Pile, and `tests/test_trr0012_transfer_analysis.py` passed its swapped-order rejection test.

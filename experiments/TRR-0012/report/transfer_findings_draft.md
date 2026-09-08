# TRR-0012 transfer findings (draft)

This bounded draft uses only the released sanitized r3 aggregate:
 /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-P11/experiments/TRR-P11/transfer/analysis_summary_r3.json,
SHA-256 8ca868aa5665d9bef5f0b9aee21c2dd9b4a3c25357ba558f8b3eb669f7deadc2. It supersedes the preserved r2 aggregate without adding a calculation. The public matrix is frozen at 14 cells with 4,064 scored positions per cell. No threshold was fitted and the summary makes no deployment claim.

## Observed accuracy

| Condition | Finance | Pile |
|---|---:|---:|
| Clean public baseline | 4,062/4,064 tokens; 30/32 exact records | 4,051/4,064 tokens; 24/32 exact records |
| Five controlled artificial variants (four prefix conditions plus the after-cut null) | unchanged from clean | unchanged from clean |
| Historical public LoRA benchmark | 4,061/4,064 tokens; 29/32 exact records | 4,048/4,064 tokens; 21/32 exact records |

The after-cut null preserved activations and predictions exactly in both domains. The historical condition has one broken Finance token event and four broken Pile token events, with one Pile improvement. It is a mandatory historical benchmark condition, not an independent target.

## Internal displacement and geometry

The four prefix perturbations measurably displaced the recorded hidden and projected features despite changing zero prediction rows in either domain. Across those four variants and domains, the reported mean relative raw-activation L2 is about 0.0030–0.0106, with reported maxima about 0.0063–0.0268; the after-cut null is exactly zero on the corresponding raw and projected feature displacement fields. This supports representation sensitivity without an observed accuracy drop; whether the displacement is material for a downstream detector remains task-dependent.

The normalized feature-shift means for the four prefix variants are about 0.0138–0.0791, with maxima about 0.871–10.15. The historical condition is substantially larger on this geometry: mean 1.322 in Finance and 2.407 in Pile, with maxima 77.30 and 349.00. These comparisons are descriptive only: the historical values come from one mandatory historical condition with five broken token events, while the artificial variants have zero broken rows.

The historical margin-normalized feature-shift AUC is near one (Finance 0.9998; Pile 0.9974), while raw activation L2 and relative raw activation L2 are much lower (Finance 0.3376 and 0.1463; Pile 0.4964 and 0.5987). All five artificial variants have UNKNOWN breakage AUC under the fixed baseline-correct risk-set definition because no broken rows are present. The AUC values are evaluator-only summaries, not thresholds, certificates, or calibration.

The released observations demonstrate stable new-source accuracy under the five controlled artificial variants and the clean baseline, while showing measurable internal displacement for the four prefix variants. They do not establish global transfer robustness or predictive calibration. Confirmation scores and larger independent held-out confirmation remain pending; the earlier r2 prose count of six artificial conditions is corrected here to five.

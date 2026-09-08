# Frozen B1 shortlists for small-budget A2 under target drift

The table below answers how often the correct token remained in B1’s top 8/16/32 at every tested snapshot. Clip coverage is the fraction of 32 clips containing every correct post-BOS token in its fixed lists; it is a perfect-selector upper bound, not achieved A2 reconstruction.

| Domain | Target stage | Token recall@8 / @16 / @32 | Whole-clip coverage@8 / @16 / @32 |
|---|---:|---|---|
| pile | 0 | 99.926% / 100.000% / 100.000% | 30/32 (93.750%) / 32/32 (100.000%) / 32/32 (100.000%) |
| pile | 64 | 99.951% / 100.000% / 100.000% | 30/32 (93.750%) / 32/32 (100.000%) / 32/32 (100.000%) |
| pile | 128 | 99.926% / 100.000% / 100.000% | 30/32 (93.750%) / 32/32 (100.000%) / 32/32 (100.000%) |
| pile | 256 | 99.926% / 100.000% / 100.000% | 30/32 (93.750%) / 32/32 (100.000%) / 32/32 (100.000%) |
| finance | 0 | 99.975% / 100.000% / 100.000% | 31/32 (96.875%) / 32/32 (100.000%) / 32/32 (100.000%) |
| finance | 64 | 100.000% / 100.000% / 100.000% | 32/32 (100.000%) / 32/32 (100.000%) / 32/32 (100.000%) |
| finance | 128 | 100.000% / 100.000% / 100.000% | 32/32 (100.000%) / 32/32 (100.000%) / 32/32 (100.000%) |
| finance | 256 | 100.000% / 100.000% / 100.000% | 32/32 (100.000%) / 32/32 (100.000%) / 32/32 (100.000%) |

Stage 0 is the unchanged public base; stage 64 is the first fitted compatible LoRA descendant; stages 128 and 256 continue that same prefix-changing trajectory. This tests one bounded adaptation from the public base, not an independently released full-SFT descendant.

**Hybrid status:** no small-budget B1+A2 hybrid was tested. The validated available A2 implementation uses untouched public model-prefix weights plus its own reconstructed-token cache. No validated maintained recovered-model-prefix path was available in the inspected/reused assets. Therefore equally accurate faster hybrid recovery and end-to-end target tracking remain untested.

The smallest empirically promising budget must lose no more than 0.1 percentage points of token recall and 2 percentage points of whole-clip coverage versus both B1/K256 and A1/K256 in every domain/stage. These limits were committed before outcomes. Passing an empirical limit does not establish equivalence.

| Domain | Stage | Smallest budget meeting both empirical loss limits |
|---|---:|---:|
| pile | 0 | 16 |
| pile | 64 | 16 |
| pile | 128 | 16 |
| pile | 256 | 16 |
| finance | 0 | 16 |
| finance | 64 | 8 |
| finance | 128 | 8 |
| finance | 256 | 8 |

Budgets meeting the empirical limits throughout the matrix: [16, 32]. No confidence fallback, retraining, recalibration, or selective stage omission was used.
For pile, the smallest empirically viable list at stages 0 → 64 → 128 → 256 was: 16 → 16 → 16 → 16.
For finance, the smallest empirically viable list at stages 0 → 64 → 128 → 256 was: 16 → 8 → 8 → 8.
Across the named cells and small budgets, B1 minus historical A1 token recall ranged from +0.886 to +3.912 percentage points. The complete separate comparisons are in the structured result; no cross-snapshot pooled score is used.

At K=8, Pile omitted 3, 2, 3 and 3 correct tokens at stages 0, 64, 128 and 256 respectively, affecting two clips each time. Finance omitted one token in one base clip and none at later snapshots. K=16 and K=32 omitted zero tokens in this panel. K=8 therefore fails the predeclared whole-clip limit despite its high token recall.

A high top-1 average also hid substantial clip errors: at the final Pile snapshot B1 top-1 accuracy was 99.237%, but only 20/32 clips were completely correct. K=8 raised this to 30/32, and K=16 to 32/32 fixed-list coverage. The smallest empirically viable shortlist did not grow along this trajectory.

The final boundary activations changed materially from base: mean per-record relative L2 was 0.348 for Pile and 0.517 for Finance. No shortlist collapse was observed through the tested updates, but this single bounded LoRA trajectory does not establish robustness to arbitrary fine-tuning.

## Complete shortlist matrix

Each domain has 32 unique paired records and 4,064 scored positions per snapshot. The entries below use K = 1 / 8 / 16 / 32 / 64 / 256 in that order. All omission counts, conditional rescue rates and uncertainty intervals are retained in `shortlist-metrics.csv` beside the structured results.

| Domain | Stage | Proposer | Token recall (%) at the six budgets | Complete clips out of 32 at the six budgets |
|---|---:|---|---|---|
| pile | 0 | b1 | 99.262 / 99.926 / 100.000 / 100.000 / 100.000 / 100.000 | 19 / 30 / 32 / 32 / 32 / 32 |
| pile | 0 | a1 | 84.916 / 96.260 / 97.982 / 98.770 / 99.311 / 99.852 | 0 / 4 / 9 / 14 / 21 / 31 |
| pile | 64 | b1 | 99.336 / 99.951 / 100.000 / 100.000 / 100.000 / 100.000 | 21 / 30 / 32 / 32 / 32 / 32 |
| pile | 64 | a1 | 85.261 / 96.531 / 97.982 / 98.745 / 99.434 / 99.828 | 0 / 4 / 10 / 14 / 24 / 29 |
| pile | 128 | b1 | 99.237 / 99.926 / 100.000 / 100.000 / 100.000 / 100.000 | 20 / 30 / 32 / 32 / 32 / 32 |
| pile | 128 | a1 | 84.916 / 96.506 / 97.958 / 98.622 / 99.459 / 99.828 | 0 / 5 / 10 / 11 / 24 / 30 |
| pile | 256 | b1 | 99.237 / 99.926 / 100.000 / 100.000 / 100.000 / 100.000 | 20 / 30 / 32 / 32 / 32 / 32 |
| pile | 256 | a1 | 84.621 / 96.014 / 97.933 / 98.597 / 99.360 / 99.803 | 0 / 5 / 9 / 12 / 20 / 29 |
| finance | 0 | b1 | 99.852 / 99.975 / 100.000 / 100.000 / 100.000 / 100.000 | 30 / 31 / 32 / 32 / 32 / 32 |
| finance | 0 | a1 | 92.569 / 97.392 / 98.401 / 98.991 / 99.582 / 99.926 | 0 / 13 / 15 / 21 / 27 / 31 |
| finance | 64 | b1 | 99.877 / 100.000 / 100.000 / 100.000 / 100.000 / 100.000 | 31 / 32 / 32 / 32 / 32 / 32 |
| finance | 64 | a1 | 93.775 / 97.392 / 98.302 / 99.065 / 99.483 / 99.951 | 5 / 13 / 15 / 22 / 25 / 31 |
| finance | 128 | b1 | 99.852 / 100.000 / 100.000 / 100.000 / 100.000 / 100.000 | 30 / 32 / 32 / 32 / 32 / 32 |
| finance | 128 | a1 | 93.725 / 97.416 / 98.327 / 99.114 / 99.483 / 99.951 | 5 / 14 / 15 / 22 / 27 / 31 |
| finance | 256 | b1 | 99.852 / 100.000 / 100.000 / 100.000 / 100.000 / 100.000 | 30 / 32 / 32 / 32 / 32 / 32 |
| finance | 256 | a1 | 93.676 / 97.392 / 98.253 / 99.040 / 99.409 / 99.951 | 4 / 14 / 15 / 22 / 27 / 31 |

## Source-level uncertainty and paired effects

The structured result retains every per-record rank (right-censored beyond 256), omission count, paired B1-minus-A1 comparison, and change from stages 0 and 64. Intervals resample source records jointly with 10,000 draws and seed 9013. Each domain/stage is reported separately.

With 32 records, even zero records losing candidate inclusion gives a one-sided 95% exact upper bound of approximately 8.94% on the population probability of any loss in a record. A zero-width bootstrap interval at zero observed losses is not evidence of equivalence. This modest pilot can identify clear omissions; it cannot establish very tight whole-clip noninferiority.

| Domain | Stage | B1 K32−A1 K256 token recall (pp; 95% paired interval) | B1 K32−B1 K256 clip coverage (pp; 95% paired interval) |
|---|---:|---|---|
| pile | 0 | +0.148 [+0.000, +0.443] | +0.000 [+0.000, +0.000] |
| pile | 64 | +0.172 [+0.000, +0.443] | +0.000 [+0.000, +0.000] |
| pile | 128 | +0.172 [+0.000, +0.492] | +0.000 [+0.000, +0.000] |
| pile | 256 | +0.197 [+0.000, +0.517] | +0.000 [+0.000, +0.000] |
| finance | 0 | +0.074 [+0.000, +0.221] | +0.000 [+0.000, +0.000] |
| finance | 64 | +0.049 [+0.000, +0.148] | +0.000 [+0.000, +0.000] |
| finance | 128 | +0.049 [+0.000, +0.148] | +0.000 [+0.000, +0.000] |
| finance | 256 | +0.049 [+0.000, +0.148] | +0.000 [+0.000, +0.000] |

## Actual boundary change

Relative L2 is computed per record over all 127 post-BOS activation vectors and averaged over records. Cosine distance averages post-BOS positions. Values are measured from paired BF16 observations cast to FP64 for this diagnostic.

| Domain | Stage | Reference stage | Mean relative L2 | Mean cosine distance |
|---|---:|---:|---:|---:|
| finance | 0 | 0 | 0 | 1.18562e-17 |
| finance | 0 | 64 | 0.421694 | 0.0940509 |
| finance | 64 | 0 | 0.427095 | 0.0940509 |
| finance | 64 | 64 | 0 | 1.65277e-17 |
| finance | 128 | 0 | 0.431844 | 0.0951034 |
| finance | 128 | 64 | 0.210499 | 0.0234939 |
| finance | 256 | 0 | 0.516625 | 0.131201 |
| finance | 256 | 64 | 0.346178 | 0.0604747 |
| pile | 0 | 0 | 0 | 1.50252e-17 |
| pile | 0 | 64 | 0.276083 | 0.0403252 |
| pile | 64 | 0 | 0.27802 | 0.0403252 |
| pile | 64 | 64 | 0 | 1.57081e-17 |
| pile | 128 | 0 | 0.259178 | 0.0348115 |
| pile | 128 | 64 | 0.16298 | 0.0139018 |
| pile | 256 | 0 | 0.347909 | 0.0608418 |
| pile | 256 | 64 | 0.274393 | 0.0384451 |

## Execution costs and reproducibility

The measured cost below is shortlist production, including deterministic full-vocabulary ranking. It is not A2 reconstruction time. No candidate simulations or prefix-maintenance operations were executed. Scientific rows use the same CPU FP32 record1 path with 2 threads; TF32 is disabled. Cells may overlap other tasks on the host, so these timings do not establish uncontended cross-system acceleration.

| Domain | Stage | Proposer | Score seconds | Rank seconds | Transfer/hash seconds | Output I/O seconds |
|---|---:|---|---:|---:|---:|---:|
| finance | 0 | b1 | 14.843 | 16.606 | 1.302 | 0.020 |
| finance | 0 | a1 | 13.667 | 16.523 | 1.347 | 0.019 |
| finance | 64 | b1 | 15.731 | 16.682 | 1.395 | 0.021 |
| finance | 64 | a1 | 12.623 | 16.635 | 1.398 | 0.016 |
| finance | 128 | b1 | 16.794 | 17.704 | 1.306 | 0.019 |
| finance | 128 | a1 | 12.249 | 18.164 | 1.391 | 0.016 |
| finance | 256 | b1 | 12.334 | 16.360 | 1.342 | 0.016 |
| finance | 256 | a1 | 11.366 | 16.238 | 1.349 | 0.016 |
| pile | 0 | b1 | 16.255 | 17.144 | 1.528 | 0.022 |
| pile | 0 | a1 | 14.642 | 17.221 | 1.546 | 0.020 |
| pile | 64 | b1 | 16.752 | 17.130 | 1.523 | 0.021 |
| pile | 64 | a1 | 14.267 | 16.983 | 1.471 | 0.018 |
| pile | 128 | b1 | 15.521 | 17.127 | 1.516 | 0.020 |
| pile | 128 | a1 | 15.134 | 17.036 | 1.418 | 0.023 |
| pile | 256 | b1 | 14.713 | 16.585 | 1.295 | 0.017 |
| pile | 256 | a1 | 13.539 | 16.518 | 1.352 | 0.020 |

Total cell wall (including loading): 556.735s. Sum of package loading: 25.340s. Observation reading/validation: 0.270s. Maximum process peak RSS: 3.317GiB. B1 record forwards:256; A1 record forwards:256; A2 candidate simulations:0.

B1 fitting was not repeated. The preserved B1 native fit previously cost 2,445.086s externally; that historical fit and its public-data preparation remain required offline resources. Shared target training/capture and backup costs are recorded by TRR-P12 and linked from this task manifest, separately from reconstruction.

Actual B1 state, readout and package code were copied and hash-verified locally; the independent Windows backup bytes were verified. All package dependencies and all 16 candidate outputs are bound. Original candidates/scores/predictions and compressed tracked copies are retained, alongside raw watchdog logs and commands. Truth is evaluator-only and first opened here after the complete matrix freeze.

B1 state: `088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706`; selected step 13000; fixed public readout: `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`. Historical A1 uses the pinned public Alpaca affine lens, not an untouched-checkpoint proposer. Its one-record full scores and stable top256 order matched the native path on the preserved fixture.

## Scope and limitations

Production observations used the inherited P11 full-sequence cut-4 prefix executor (BF16, B8 × 192 with 128 active positions). Before release, its first 128 hidden states exactly matched the full 16-layer backbone on both base and adapted smoke fixtures. Production did not execute all 16 layers; the capture-profile clarification and independent equivalence receipt are preserved.

This is a paired shortlist component study on one target trajectory and 64 unique source clips shared with Agent 2. It is not an independent replication of Agent 2. Sources exclude known decoder fitting/selection, prior opened panels, Agent 1 reservations, and this trajectory’s target fitting/validation sources through the bound evaluator ledger. Unknown overlap with a public model’s original pretraining corpus is not ruled out.

The complete dual-canonical reconstruction matrix was not run, and no new active reconstruction replacement is registered. No result here is an overall-best, canonical-replacement, equally accurate hybrid, or end-to-end tracking claim. A negative shortlist budget is a completed Stage1 finding. A positive shortlist budget would only justify a separate maintained-prefix integration study with reserved confirmation sources.

Reproduction: `experiments/agent3-b1-small-budget-a2/README.md`. Structured evidence: `experiments/agent3-b1-small-budget-a2/manifest.json`. Prospective decisions: `experiments/agent3-b1-small-budget-a2/plan.md`. Failures and repairs are retained in validation receipts; no scientific output was selected from an excluded attempt.

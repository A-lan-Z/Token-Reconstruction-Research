## Summary

TRR-0010 completes the frozen four-cell public evaluation of the current and expanded fitting banks with fixed and learned readouts, plus the retained frozen A1+A2 comparator. The public prediction matrix contains all 24 method-by-cell outputs and passed the public gate before the single truth-loading phase.

The public inference and gate phase is bound to code commit `d493e16ac8237ad835b4459c419c4ae55b00f534`. Final scored evidence and the report are committed at `e3e92931bcb5090e49ea373b247a8098e3e246b1`, separately from the frozen inference phase.

## Results

- The expanded fixed readout is the strongest fitted fixed-bank result on every cell: 99.4156–99.5017% token accuracy and 87–98 exact clips out of 128.
- The learned expanded directional readout is lower on every cell: 98.2099–99.2311% token accuracy and 32–84 exact clips out of 128.
- The retained A1+A2 comparator is higher on the same four-cell panel: 99.6248–99.8339% token accuracy and 108–114 exact clips out of 128. It remains a historical comparator and is not presented as a newly fitted replacement.
- The route-specific decision readout is `CHECKS_COMPUTED_NO_POOLED_DECISION`: the data route (`expanded_fixed`) is `PASS` for both Finance and Pile, while the directional route is `FAIL` for both domains because the learned directional candidate does not pass the registered candidate-versus-expanded-fixed quality route.
- All registered route-specific cost gates pass. Warmed candidate/A1+A2 runtime ratios are 0.00475–0.00500 across cells; expanded-directional versus expanded-fixed warmed runtime ratios are 0.990–1.039 with GPU peak ratio 1.0. Warmed runtime excludes shared preparation.
- B0/B1 frequency strata were scored, but the registered readout keeps rare/absent strata as `DIAGNOSTIC_ONLY`; they do not create a route-level pass or replacement claim. The minimum observed exposed-source count in the scored bins was 38.

These results do not claim a pooled overall winner, canonical-complete reconstruction, or replacement of the retained comparator.

## Validation and evidence

- Focused validation receipts: the current metadata-only curator adapter suite reports 3 passed in 1.59s (`experiments/TRR-0010/evaluation/curator_adapter_test_receipt_r5.json`); the latest planning scorer-correction suite reports 53 passed in 5.90s (`experiments/TRR-0010/planning/scorer_safeguard_domain_correction_receipt_v1.json`). The separate score-CLI synthetic accounting reports 13 tests; these are distinct suites and are not added together.
- Public predictions: `experiments/TRR-0010/evaluation/public_prediction_watchdog_r5/run_manifest.json` (SHA256 `d239fb03e1b0f5ec99e23f2482b1ba4bf7d5c3d8db852f9a6b999f077de396f5`), status `PUBLIC_PREDICTIONS_COMPLETE_BEFORE_TRUTH`.
- Frozen score: `experiments/TRR-0010/evaluation/score_v1.json` (988,310 bytes, SHA256 `69e70215c4da85957117a6cbb350b31bcba36f754ead026d05bb0d1157651c49`).
- Score boundary receipt: `experiments/TRR-0010/evaluation/score_v1.boundary.json` (SHA256 `6b81861013d8c3bad8a76d629c43fc942f28d0a1b0189527279b5f34495e8d0d`).
- Public timing/cost evidence: `experiments/TRR-0010/evaluation/cost_evidence_r5.json` (84,652 bytes, SHA256 `cedd7c14fbec2f8aaf45e6e34a7a0da865bf7e78adc7947db249f9b08ba5c049`), status `COST_EVIDENCE_COMPLETE_PUBLIC_TIMING`.
- Public-development learning-curve figure: `experiments/TRR-0010/report/learning_curve_public_development_v1/`, generated only from the public diagnostics table bound in its receipt.

Human-readable result: `coordination/results/TRR-0010.md`. Structured evidence: `experiments/TRR-0010/manifest.json`.

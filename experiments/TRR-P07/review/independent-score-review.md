# TRR-P07 independent score review

This is a metadata-only audit of experiments/TRR-P07/runtime/scored-r2/results.json.
Verified result SHA256: f60f71c3ca747c6120671458bb54e559764c462a21abf9845d99967d13c76a38. No rescoring, rerun, selection, or raw-truth read was performed.

## Primary contrast

The registered primary is past_minus_reference. All 8 cell-level token 95% intervals have lower endpoints above zero; the minimum lower endpoint is 0.053826280 percentage points.
All 16 primary per-seed rows are positive, with 32,512 scored tokens and 256 exact-record opportunities per seed.

| cell | token delta pp | token 95% CI pp | exact delta pp | exact 95% CI pp | gate cell |
| --- | ---: | --- | ---: | --- | --- |
| p06_panel/finance/public_base | 0.126107 | [0.053826, 0.203002] | 3.125000 | [-0.585938, 6.835938] | uncertain |
| p06_panel/finance/public_lora_2601 | 0.141486 | [0.069205, 0.219919] | 5.273438 | [1.562500, 9.375000] | support |
| p06_panel/pile/public_base | 0.712045 | [0.489050, 0.939653] | 2.148438 | [0.000000, 4.492188] | uncertain |
| p06_panel/pile/public_lora_2601 | 0.873524 | [0.644377, 1.114973] | 1.757812 | [-0.781250, 4.492188] | uncertain |
| trr0006_subset/finance/public_base | 0.164555 | [0.064592, 0.264518] | 7.421875 | [3.320312, 11.523438] | support |
| trr0006_subset/finance/public_lora_2601 | 0.346026 | [0.244525, 0.445989] | 7.421875 | [3.515625, 11.718750] | support |
| trr0006_subset/pile/public_base | 0.676673 | [0.505967, 0.853531] | 1.171875 | [-0.390625, 2.929688] | uncertain |
| trr0006_subset/pile/public_lora_2601 | 0.713583 | [0.530573, 0.898130] | 2.539062 | [0.781250, 4.687500] | uncertain |

## Arithmetic and gate checks

The fractional gain/loss and both/neither partitions passed for all 32 frozen cell-by-contrast rows (8 cells × 4 contrasts), with aggregate point arithmetic also passing.
The registered primary gate recomputes exactly: 3 supported cells, 0 harm cells, support in both panels, but false domain-target coverage because neither pile target has a supported cell. The resulting disposition is PANEL_DEPENDENT_OR_UNCERTAIN and automatic follow-on is false.
Positive effects and positive confidence intervals therefore do not by themselves satisfy the utility gate.

## Claim boundary

No prior-panel percentage comparison, cross-panel improvement percentage, or pooled panel/domain/target percentage is claimed. Reporting remains cellwise and source-cluster paired, as required by the frozen plan.

Audit status: PASS. The scored result opened truth only after prediction freeze; this review did not read or persist raw truth.

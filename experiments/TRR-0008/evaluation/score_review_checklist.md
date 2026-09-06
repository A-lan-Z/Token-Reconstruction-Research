# TRR-0008 independent score review checklist

Status: **ready for the post-score review; no truth sidecar or score artifact was opened here**. Apply this checklist after the owner-authorized public freeze and score pass. The sole numeric source is `experiments/TRR-0008/planning/decision_contract.json` with status `FROZEN_DECISION_CONTRACT_BEFORE_SOURCE_SELECTION` and SHA-256 `141a3a506e20e1eadd3be1beb53c2f01508e9b651c93d19b2f3d9ff72344bcf3`.

## Public freeze before truth

- [ ] Run `scripts/trr0008_eval_gate.py` with the registered prediction run manifest and `experiments/TRR-0008/timing/precision40_result.json`; require `PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH`.
- [ ] Confirm the registration, run manifest, observation manifest, normalized public embedding, timing receipt, and freeze receipt each match their recorded path, byte count, and SHA-256. The timing binding must be `experiments/TRR-0008/timing/precision40_result.json`, schema `token-reconstruction.trr0008-balanced-timing.v1`, 40 blocks, `qualification=PASS`, and `truth_opened=false`.
- [ ] Confirm the matrix is exactly the four scientific methods in contract order: `trr6__enriched_trained_diagonal_attention128` (reference), `current_enriched__residual_mlp512` (credible alternative), `improved_public_bank__residual_mlp512` (candidate), and `improved_public_bank__trained_diagonal` (diagnostic).
- [ ] Confirm all four cells are present: `finance__public_base`, `finance__public_lora_2601`, `pile__public_base`, and `pile__public_lora_2601`; counts are 1,024 per Finance target and 384 per Pile target, with 127 scored post-BOS positions per row.
- [ ] Confirm every prediction and timing artifact has the expected geometry, hash, method/cell identity, synchronized boundary, and warmup/measured exact-output match. Reject missing, duplicate, alternate-root, changed-code, or changed-registration artifacts.
- [ ] Confirm the truth binding header was produced only after the public gate, remains metadata-only before scoring, points to a sidecar outside the repository and prediction root, and carries the observation record-ID digest and registered counts.

## Numeric score recomputation

- [ ] Load the score JSON only after the public gate and truth-binding checks. Require JSON serialization with no tensors, non-finite values, or undeclared fields that affect decisions.
- [ ] Require bootstrap `seed=8008`, `draws=10000`, `unit=source_record`; token positions remain nested within source records. Recompute record-level token means, never treating the 127 positions as independent records.
- [ ] For each candidate/reference cell, recompute paired exact gains and losses. Primary exact bounds use Clopper–Pearson component alpha `0.0125` for gain lower and loss upper; the exact lower bound is gain lower minus loss upper. Safeguard exact bounds use component alpha `0.025`.
- [ ] Recompute paired token record means with the declared bootstrap streams and one-sided alphas: primary `0.025`, safeguards `0.05`. Compare the recomputed lower endpoints with `token_net_bootstrap_975.lower` and `token_net_bootstrap_95.lower` in the score artifact.
- [ ] Primary cell is `finance__public_base`; contrast is candidate minus frozen reference. A reliable positive route requires exact LCB `> 0` or token LCB `> 0`. A practical advance requires exact LCB `>= +0.05` or token LCB `>= +0.01`. Point estimates do not clear either gate.
- [ ] Check all four candidate/reference safeguard cells. Every exact lower bound must be `>= -0.05`, and every token lower bound must be `>= -0.01`; report each cell separately and do not pool domains, targets, methods, or token positions.
- [ ] Check descriptive contrasts for the current residual and improved diagonal without using them as pooled winners or alternate candidate selection.

## Cost and final decision

- [ ] Confirm the scorer sees the bound 40-block timing receipt and obtains `cost_status=COST_PASS`; every cell is required, including the primary Finance cell, and each valid per-cell upper ratio bound is `<= 1.25`. An invalid or inconclusive timing binding retains the reference.
- [ ] Confirm the final status is interpreted literally: only `PROMOTE_PRIMARY_CANDIDATE` permits promotion. `UNRESOLVED_REFERENCE_RETAINED`, `RELIABLE_BUT_PRACTICAL_MAGNITUDE_UNCERTAIN`, `NO_PROMOTION_SAFEGUARD_FAILURE`, `COST_EVIDENCE_MISSING`, `COST_EVIDENCE_INVALID`, `COST_EVIDENCE_INCONCLUSIVE`, and `COST_GATE_FAILED` retain the reference.
- [ ] Confirm the primary harm safeguard is reported separately even though the primary cell is also a quality endpoint.
- [ ] Record the score, decision, freeze, truth-binding, sidecar, prediction, and timing hashes, the exact scorer command, start/end times, and the post-score artifact path. Do not refit, reroute, expand the panel, swap models, or alter timing after truth access.

## Independent arithmetic receipt

The reviewer should write a compact comparison containing: method/cell matrix completeness; per-cell record and token counts; primary exact/token lower bounds and margins; four safeguard exact/token lower bounds and margins; timing status and per-cell upper ratios; scorer status/promotion; and any hash or serialization mismatch. A mismatch in a binding or endpoint is an integrity failure requiring a fail-closed result, not a repaired score.

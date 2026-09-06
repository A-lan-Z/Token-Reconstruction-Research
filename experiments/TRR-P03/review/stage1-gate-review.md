# TRR-P03 Stage 1 gate review

## Decision

**STOP_VARIANT; Stage 2 was not run because the predeclared Stage 1 gate failed.** The result is a clear regression on this natural-text panel against historical A1 in both target arms. This is an exploratory task decision for this frozen panel, not a universal claim about every projection or compression method.

## Predeclared gate

The score files use 10,000 length-stratified paired record-cluster bootstrap draws with seed 20260905 and six records per length stratum. The matched public arm has projected 928/1482 correct tokens versus A1 1332/1482; its projected minus A1 delta is -27.2605 percentage points, with 95% CI [-32.7935, -20.1754]. The shifted arm is 922/1482 versus 1333/1482, delta -27.7328 percentage points, CI [-33.0634, -20.9852].

All five predeclared tests fail:

- Matched projected minus A1 at least +1 percentage point: observed -27.2605 pp — FAIL.
- Matched paired record-cluster CI lower bound above 0: observed -32.7935 pp — FAIL.
- Shifted projected minus A1 nonnegative: observed -27.7328 pp — FAIL.
- Matched exact-record count not lower: projected 0 versus A1 5 — FAIL.
- Shifted exact-record count not lower: projected 0 versus A1 5 — FAIL.

Exact-record CIs are reported descriptively in gate.json, not used as an extra ambiguous gate. The matched exact-record delta is -20.8333 pp with CI [-33.3333, -8.3333].

## Paired changes and secondary context

At token positions, the matched arm has 91 projected gains and 495 regressions (net correct-token change -404); the shifted arm has 90 gains and 501 regressions (net -411). At the record level, projected accuracy is lower on 22 of 24 records in each arm, with median delta -34.3750 pp and worst delta -43.5897 pp. Macro deltas are -30.2192 pp matched and -30.5864 pp shifted.

Projected still beats raw: it exceeds raw by 430 tokens (29.0148 pp) in the matched arm and by 443 tokens (29.8920 pp) in the shifted arm. The four-record A1+A2 anchor remains an accuracy anchor at 100% over 156 scored positions in both arms; its limited coverage does not rescue the full-panel gate.

The P02 tiny endpoint advantage may reflect narrow tuple geometry or selection. The broader natural, varied-length diagnostic shows that this static projected variant does not preserve A1 accuracy, but the comparison cannot identify that mechanism from these data alone.

## Reproducibility

The compact machine-readable gate is in gate.json. Its inputs are the frozen metrics, paired-statistics, scoring-evidence, pre-score, and strict-validation receipts, each recorded with byte count and SHA-256, and the source commit is 6edb276a3a536988a1d2cc9f3aa4c29e90e1a6b1. No Stage 2 observations or truth were opened.

Next decision: deprioritize this static projected variant, retain A1+A2 accuracy anchor reporting, and do not automatically repeat the same compression.

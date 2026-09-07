# TRR-P08 public-fit review

Status: **PASS_PUBLIC_FIT_ONLY_PENDING_FREEZE**.

I reviewed the completed `main-r2` receipt and watchdog records without opening
fresh natural-panel truth or changing selection. The matrix contains four arms
at each of seeds 6106 and 6107: positionwise/past-only crossed with
joint/staged. Every arm reports `PASS`, 3000 updates, and the registered H128,
record-batch-8, 512-draw geometry. The run used the common public fit manifest,
full-vocabulary validation token accuracy at step 0 and every 100 updates, and
one seed-specific ordered schedule shared by all four arms at that seed.

## Matrix and selection

| seed | arm | schedule | selected step | best validation token accuracy | final validation token accuracy | exact final / 48 |
|---:|---|---|---:|---:|---:|---:|
| 6106 | positionwise | staged | 2200 | 0.961771 | 0.960429 | 17 |
| 6106 | past-only | staged | 1200 | 0.962106 | 0.960429 | 16 |
| 6106 | positionwise | joint | 1500 | 0.964453 | 0.962441 | 17 |
| 6106 | past-only | joint | 1500 | 0.964453 | 0.962106 | 18 |
| 6107 | positionwise | staged | 1900 | 0.964118 | 0.962106 | 15 |
| 6107 | past-only | staged | 2300 | 0.963112 | 0.962777 | 16 |
| 6107 | positionwise | joint | 1800 | 0.965459 | 0.965124 | 16 |
| 6107 | past-only | joint | 2100 | 0.964453 | 0.964453 | 18 |

Selection is consistent with the registered earliest-maximum rule. The staged
arms all use 1000 affine-only updates followed by 2000 complete-model updates;
joint arms use 3000 complete-model updates. The two per-seed schedule hashes
are shared across all four arms, and all arms within a seed share the same
standard initialization and initial-state hash. No selected step or arm was
changed after observing these development curves.

## Affine transition and correction diagnostics

The affine transition is competent but not perfect on the public development
material. Seed 6106 has 139 fit errors among 112,825 valid post-BOS positions
(0.998768 accuracy) and 125 validation errors among 2,982 positions (0.958082).
Seed 6107 has 116 fit errors (0.998972) and 122 validation errors (0.959088).
The full validation transition cohorts are complete for both seeds.

The fixed 256-error-per-fit-cohort quota is short in both seeds: 139 rows for
6106, with bin shortfalls 51, 42, 0, 24; and 116 rows for 6107, with shortfalls
58, 52, 6, 24 in bins `[1-15]`, `[16-39]`, `[40-79]`, `[80-127]`. This is the
predeclared nonfatal diagnostic shortfall; it does not invalidate the fits or
permit cohort changes.

For the staged arms, the selected/final validation transition comparisons are
reported as correction gains/regressions (same-checkpoint affine-only versus
full decoder):

| seed | arm | selected | final |
|---:|---|---:|---:|
| 6106 | positionwise staged | 7 / 4 | 7 / 3 |
| 6106 | past-only staged | 5 / 0 | 5 / 4 |
| 6107 | positionwise staged | 11 / 1 | 8 / 0 |
| 6107 | past-only staged | 2 / 0 | 3 / 1 |

The fit-side counts are sparse and cohort-limited: positionwise staged is
`0/4` then `0/0` for seed 6106 and `0/1` then `0/0` for seed 6107; past-only
staged is `1/0` then `1/0` for seed 6106 and `0/0` then `0/0` for seed 6107.
These diagnostics describe the public development path only. The runner also
stores comparisons for joint arms against the shared staged transition cohort;
those are collateral same-cohort comparisons and are not interpreted as
staged-transition evidence.

## Resources and provenance

The fit receipt covers 2026-09-07T01:39:51Z through 01:52:24Z. The watchdog
finished with status `PASS`, child and wrapper exit code 0, no termination
reason, 1,496 resource samples, peak group RSS 7,569,088,512 bytes, and minimum
sampled host availability 17,729,118,208 bytes. Per-arm peak CUDA reserved
memory was at most 3,202,351,104 bytes, with at least 12,388,925,440 bytes
free; the largest per-process RSS was 5,621,690,368 bytes. These remain below
the declared 6 GiB reserved, 8 GiB free, 16 GiB RSS, and 10 GiB host-available
limits.

The child started from source commit `00504aac40f2b3758c65c14c6dcc8f1b4c6064e5`;
the final receipt was written at `ef925c550e39ea557f9fea516d9d21473a9eb163`.
The only intervening committed files are the selected-state freeze helper and
its focused test. The fit implementation, metrics, freeze, scorer, and truth
adapter source blobs are identical at both commits. This is a source-level
provenance pass; the final manifest should retain the two commit IDs and the
implementation's dependency-environment check.

The main-fit receipt SHA-256 is
`44b2a965c92d64938e5ff322222678c8cfaf4ba84ff93b62d6f49cfce39d7a86`; the
watchdog finish SHA-256 is
`5905f30bf3e29922082312679d3496714bcafd04a2eac66eac0284728f7a6695`.
The receipt records `source_token_access=false`, `target_truth_access=false`,
`guessed_token_feedback=false`, and zero candidate simulations.

## Review disposition

The public fitting matrix is complete and suitable for selected-state freeze.
This review makes no quality claim about the fresh natural panel and does not
open or score evaluator truth. Fresh source selection, prediction freezing,
truth materialization, and scoring remain separate authorized gates.

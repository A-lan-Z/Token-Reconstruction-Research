# TRR-P08 staged affine-first versus joint fitting

Status: **COMPLETE — fresh capture, prediction matrix, post-freeze truth materialization, and fixed scoring passed.** The task-local contextual gate is `USEFUL_CONTEXTUAL_BENEFIT_RULED_OUT`; the separately registered general-staging gate is `GENERAL_STAGING_INCONCLUSIVE`. These are exploratory P08 decisions on the fixed H128 panel and do not establish a universal mechanism or replace the canonical benchmark.

## Decision

On the primary `public_base` target, the staged-versus-joint interaction is positive for Pile at +0.2153 percentage points (95% source-record cluster CI [+0.0277, +0.4045]) and near zero for Finance at −0.0123 pp (CI [−0.0846, +0.0584]). Both upper endpoints are below the registered +0.5 pp practical token margin; the exact-recovery interaction is also far below the registered +5 pp margin in both domains. No seed reaches either practical benefit margin. The predeclared deterministic interaction gate therefore rules out useful contextual benefit for this fixed comparison, while the general staged-minus-joint effects remain a separate inconclusive question.

The next decision is to deprioritize this tested affine-first staging schedule against the joint/reference arms; no promotion follows. Any later test would require a separately planned hypothesis and evidence; this result does not claim that every possible staged optimizer schedule lacks benefit.

## Frozen scope and analysis

The study compares positionwise visibility (`j=i`) and past-only visibility (`j<=i`) under equal total budgets:

| visibility | joint | staged |
|---|---:|---:|
| positionwise | 3000 complete-model updates | 1000 affine-only + 2000 complete-model updates |
| past-only | 3000 complete-model updates | 1000 affine-only + 2000 complete-model updates |

There are two paired fit seeds (6106, 6107), eight public fits, common H128 geometry, common ordered record and position draws, standard identity affine initialization (`W=I`, `b=0`, `s=3.0`), and 256 fresh records per domain (Pile and Finance), with 127 scored post-BOS positions per record. `public_base` is primary; `public_lora_2601` is a paired transfer diagnostic and does not select an arm or enter the primary gate.

The primary interaction is

```text
I = (Past_staged − Positionwise_staged)
    − (Past_joint − Positionwise_joint).
```

Correctness counts, exact indicators, and paired gains/losses were averaged within source record across the two seeds before resampling. Seeds are replicates rather than source records. All intervals below use 10,000 paired source-record cluster bootstrap draws, seed 8080, with the same source-index schedule reused across methods and paired targets within each domain. Token intervals are percentage points; exact intervals use the 256-record denominator.

The contextual support gate required +0.5 pp token or +5 pp exact interaction benefit with a positive lower CI endpoint, no interaction harm, and a same nonnegative interaction sign across seeds within each primary domain. Useful benefit was ruled out only when both primary-domain interaction upper endpoints were below both practical margins and no seed reached either registered benefit margin. General staging was assessed separately for each visibility mask and was not folded into the contextual decision.

## Source binding and preserved deviation

Before capture, a metadata-only audit found that the effective r1 exclusion collector omitted the inherited approved TRR-0007 opaque ledger. A distinct approved TRR-0009 export arrived later and was also bound during repair; the original r1 panel was excluded before capture. Its receipt is `experiments/TRR-P08/runtime/source-selection-r1/selection.json` (SHA-256 `75f8d66007d6ae5558ded8b0fc428fc88db8aa5a04f4d372bd187f5934079274`). Comparing the r1 panel with the two ledgers found 0 source-level and 1 H128-sequence overlap with each, for 2 unique H128 overlaps. No observation, model/target, prediction, or truth payload was opened during that audit.

The repaired r2 universe and selection bound all four required opaque ledgers, including the original and replacement TRR-0009 exports, kept the same 512-record panel and seed 8108, and passed the zero-intersection audit. Two Finance records changed and 510 records were retained; Pile was unchanged. The repaired selection was used for the r3 public capture and later truth binding:

- universe: `experiments/TRR-P08/runtime/source-universe-r2/frozen.json`, SHA-256 `23a8c632e232552e3dd5a33bcb1d5b07387505a971fb6c7df55ab1f8d5316a72`;
- selection: `experiments/TRR-P08/runtime/source-selection-r2/selection.json`, SHA-256 `e8023ed00ae3efbaa1b86f1290aa4de6b23da48dd70f7b10d9e147fbf6fbfb86`;
- disjointness audit: `experiments/TRR-P08/runtime/source-selection-r2/selection-disjointness-audit-r2.json`, SHA-256 `1376eac05969d2708de0b6031993b0a167482a415620faecee9248694e65a9f2`.

## Public fit qualification

All eight fits passed 3000 updates with the registered schedules. Validation selection used full-vocabulary token accuracy at step 0 and every 100 updates with earliest maximum selection.

| seed | arm | selected step | best validation token accuracy | final validation token accuracy | final exact / 48 |
|---:|---|---:|---:|---:|---:|
| 6106 | positionwise staged | 2200 | 0.961771 | 0.960429 | 17 |
| 6106 | past-only staged | 1200 | 0.962106 | 0.960429 | 16 |
| 6106 | positionwise joint | 1500 | 0.964453 | 0.962441 | 17 |
| 6106 | past-only joint | 1500 | 0.964453 | 0.962106 | 18 |
| 6107 | positionwise staged | 1900 | 0.964118 | 0.962106 | 15 |
| 6107 | past-only staged | 2300 | 0.963112 | 0.962777 | 16 |
| 6107 | positionwise joint | 1800 | 0.965459 | 0.965124 | 16 |
| 6107 | past-only joint | 2100 | 0.964453 | 0.964453 | 18 |

The matrix used 12,288,000 public query draws. Summed arm wall time was 750.484 s; the watchdog interval was 756.123 s. The largest fit recorded 3,202,351,104 bytes reserved CUDA memory, 12,388,925,440 bytes free, and 5,621,690,368 bytes process RSS, within the approved limits.

The public affine transition was competent but not perfect: seed 6106 had 139 fit errors among 112,825 valid positions and 125 validation errors among 2,982 positions; seed 6107 had 116 fit errors and 122 validation errors. The registered 256-error fit quota was short, so the transition diagnostic is informative but bounded; no cohort was expanded or selected after seeing fresh answers. On the complete validation cohorts, full-versus-same-checkpoint-affine correction gains/regressions were 7/3 and 5/4 for seed 6106 positionwise and past-only final states, and 8/0 and 3/1 for seed 6107.

## Capture, freeze, truth, and scoring evidence

The repaired public capture passed with child time 14.274 s and watchdog time 16.220 s. The prediction run produced all 32 source-free prediction descriptors in 170.180 s. Across the descriptor timing records, measured passes summed to 121.669 s, warmup to 40.989 s, and observation loading to 4.994 s; maximum prediction CUDA reserved memory was 2,661,285,888 bytes and maximum process RSS was 3,615,170,560 bytes. Each selected state was 20,991,156–20,991,188 bytes (about 20.99 MB), and the common normalized public-E asset was 1,050,673,488 bytes. The three-pass mean per 256-record arm, averaged over both targets, domains, and seeds, was 1.2701 s for past/joint, 1.2595 s for past/staged, 1.2593 s for positionwise/joint, and 1.2806 s for positionwise/staged; the corresponding observed ranges across the eight target/domain/seed cells were 1.2472–1.3130, 1.2330–1.2942, 1.2426–1.3176, and 1.2494–1.3179 s. These are warmed batch-8 throughput measurements, not single-record latency.

The create-only joint freeze is `experiments/TRR-P08/runtime/joint-freeze-r3/joint_freeze.json` (SHA-256 `4de6ee0986b6f62ec64073bed87d23fe899c599ccf9b333e0114964e7579ba9e`, status `JOINT_FREEZE_VALIDATED_NO_TRUTH`). Its timestamped provenance is `experiments/TRR-P08/runtime/joint-freeze-r3/joint_freeze_provenance.json` (SHA-256 `1823e8d19e96bb8e2bd406bf3dea448a42f92f6a616f793353e858252ea11f52`) and records 32 predictions, 8 states, the exact command, plan hash, observation hash, and `truth_opened=false`. The clarification receipt `experiments/TRR-P08/runtime/joint-freeze-r3/joint_freeze_provenance_clarification.json` (SHA-256 `b4b094050d92c3790b9eeb84028b3a889ad0136b59f151bc01593a5c438e51fd`) states that the provenance creation/finish times are metadata-recording times, not timestamps for the earlier `freeze_matrix.py` execution.

After root authorization, the truth materializer wrote only to `/tmp/trr-p08-evaluator-truth-r3`; its manifest is outside the repository at `/tmp/trr-p08-evaluator-truth-r3/truth.manifest.json` (SHA-256 `c31e681e814cd093369f43faa846c2b316dea717a1213dbc4edea28f9dcbb679`). Materialization took 3.964 s and bound freeze SHA `4de6ee…` and selection SHA `e8023e…`. A metadata-only copy is retained at `experiments/TRR-P08/review/truth-manifest-r3.json` with the same SHA; the token arrays remain outside the repository. The scorer passed in 5.651 s with result `experiments/TRR-P08/runtime/scored-r3/results.json` (SHA-256 `cf1c62672f5bbddd804451e0fd129fed043ea4b7026a979e44dbbfb9770aa7d8`) and execution receipt `/tmp/trr-p08-score-execution-r3.json` (SHA-256 `ca7db6bc13e4682860b1bb362497a8e195eff81c12e5e76729d8785ba2dfe463`). A copy is retained at `experiments/TRR-P08/review/score-execution-r3.json`.

A preserved r2 capture attempt also failed closed before model load because the inherited validator expected `panel.CANDIDATE_RANGES`; `experiments/TRR-P08/setup/capture-failure-r2.json` (SHA-256 `8ce612f63b2b657e2420dd0fd78db1a41d00dd8167b257c2d7674a37ef990b81`) records no model load, source-row materialization, observation, prediction, or truth access. The adapter was repaired before the create-only r3 capture. The first authorized truth-materialization attempt then failed closed before opening truth because the frozen selector stores `truth_opened=false` under `access_boundary`; it produced no truth output. Commit `b5e115c` added a fail-closed schema adapter that accepts the nested representation only when the resolved value is explicitly false; its focused synthetic test passed 4/4. The successful rerun used the same freeze, selection, universe, and plan bindings.

The exact successful commands are retained in `experiments/TRR-P08/review/score-plan-r3.json`, including the r3 paths, the `--output-dir` truth flag, and the scorer's `--truth-manifest`/`--output` flags. The fail-closed executable wrapper is `scripts/trr_p08/score_run_r3.py`; its exact executed copy is tracked alongside the plan. No prediction, fit, selection, or freeze artifact was altered after the freeze.

## Fresh-panel absolute quality

Each cell has 256 records and 32,512 scored tokens. The table reports each seed and their replicate average; exact recovery is the percentage of the 256 records whose 127 scored tokens are all correct.

| domain | target | method | seed 6106 token/exact % | seed 6107 token/exact % | replicate-average token/exact % |
|---|---|---|---:|---:|---:|
| pile | `public_base` | past / joint | 92.4397 / 1.5625 | 92.5504 / 1.5625 | 92.4951 / 1.5625 |
| pile | `public_base` | past / staged | 92.5197 / 1.1719 | 92.4489 / 1.5625 | 92.4843 / 1.3672 |
| pile | `public_base` | positionwise / joint | 92.5966 / 1.1719 | 92.6058 / 1.5625 | 92.6012 / 1.3672 |
| pile | `public_base` | positionwise / staged | 92.4397 / 1.5625 | 92.3105 / 1.1719 | 92.3751 / 1.3672 |
| finance | `public_base` | past / joint | 98.9604 / 43.7500 | 98.9788 / 45.7031 | 98.9696 / 44.7266 |
| finance | `public_base` | past / staged | 98.9142 / 43.3594 | 98.9727 / 46.0938 | 98.9435 / 44.7266 |
| finance | `public_base` | positionwise / joint | 98.9881 / 44.1406 | 99.0004 / 46.0938 | 98.9942 / 45.1172 |
| finance | `public_base` | positionwise / staged | 98.9635 / 44.5312 | 98.9973 / 44.9219 | 98.9804 / 44.7266 |
| pile | `public_lora_2601` | past / joint | 92.8888 / 1.1719 | 92.7288 / 1.9531 | 92.8088 / 1.5625 |
| pile | `public_lora_2601` | past / staged | 92.4766 / 1.1719 | 92.5320 / 1.1719 | 92.5043 / 1.1719 |
| pile | `public_lora_2601` | positionwise / joint | 92.5197 / 0.3906 | 92.6120 / 1.9531 | 92.5658 / 1.1719 |
| pile | `public_lora_2601` | positionwise / staged | 92.4213 / 0.7812 | 92.2890 / 1.5625 | 92.3551 / 1.1719 |
| finance | `public_lora_2601` | past / joint | 99.0219 / 45.3125 | 99.0465 / 46.8750 | 99.0342 / 46.0938 |
| finance | `public_lora_2601` | past / staged | 98.9604 / 42.5781 | 99.0004 / 47.6562 | 98.9804 / 45.1172 |
| finance | `public_lora_2601` | positionwise / joint | 99.0004 / 45.7031 | 99.0434 / 48.4375 | 99.0219 / 47.0703 |
| finance | `public_lora_2601` | positionwise / staged | 99.0188 / 47.2656 | 99.0127 / 46.4844 | 99.0157 / 46.8750 |

## Interaction and staging contrasts

The primary and transfer interaction estimates are:

| domain | target | interaction token Δ pp (95% CI) | interaction exact Δ pp (95% CI) |
|---|---|---:|---:|
| pile | `public_base` | +0.2153 [+0.0277, +0.4045] | -0.1953 [-1.5625, +1.1719] |
| finance | `public_base` | -0.0123 [-0.0846, +0.0584] | +0.3906 [-2.5391, +3.5156] |
| pile | `public_lora_2601` | -0.0938 [-0.2830, +0.0861] | -0.3906 [-1.7578, +0.9766] |
| finance | `public_lora_2601` | -0.0477 [-0.1307, +0.0292] | -0.7812 [-3.7109, +2.1484] |

On `public_base`, the Pile interaction is driven by a positionwise staged-versus-joint loss of −0.2261 pp (paired token gains/losses 235.5/309.0), while the past-only contrast is −0.0108 pp (246.5/250.0). Finance has smaller losses in both visibility masks. This explains why the Pile interaction point is positive while remaining below the practical margin: it is a relative contrast among small changes, not a useful absolute improvement claim.

General staged-minus-joint estimates, kept separate from the contextual interaction gate, are:

| domain | target | visibility | staged − joint token Δ pp (95% CI) | staged − joint exact Δ pp (95% CI) |
|---|---|---|---:|---:|
| pile | `public_base` | past-only | -0.0108 [-0.1630, +0.1292] | -0.1953 [-1.1719, +0.5859] |
| pile | `public_base` | positionwise | -0.2261 [-0.3737, -0.0892] | +0.0000 [-0.7812, +0.7812] |
| finance | `public_base` | past-only | -0.0261 [-0.0769, +0.0231] | +0.0000 [-2.3438, +2.3438] |
| finance | `public_base` | positionwise | -0.0138 [-0.0646, +0.0415] | -0.3906 [-2.5391, +1.5625] |
| pile | `public_lora_2601` | past-only | -0.3045 [-0.4614, -0.1646] | -0.3906 [-1.5625, +0.5859] |
| pile | `public_lora_2601` | positionwise | -0.2107 [-0.3445, -0.0830] | +0.0000 [-0.5859, +0.5859] |
| finance | `public_lora_2601` | past-only | -0.0538 [-0.1061, -0.0062] | -0.9766 [-3.5156, +1.3672] |
| finance | `public_lora_2601` | positionwise | -0.0062 [-0.0569, +0.0477] | -0.1953 [-2.7344, +2.3438] |

The general-staging gate is `GENERAL_STAGING_INCONCLUSIVE`; its CI and practical-margin rule does not support a claim that staging is globally beneficial or globally useless. Transfer cells show larger negative token deltas for staged past-only fitting (Pile −0.3045 pp, Finance −0.0538 pp), but these are transfer diagnostics and do not alter the primary decision.

Paired component counts for the staged-minus-joint comparisons are:

| domain | target | visibility | token Δ pp | exact Δ pp | token gains / losses |
|---|---|---|---:|---:|---:|
| pile | `public_base` | past-only staged vs joint | -0.0108 | -0.1953 | 246.5 / 250.0 |
| pile | `public_base` | positionwise staged vs joint | -0.2261 | +0.0000 | 235.5 / 309.0 |
| finance | `public_base` | past-only staged vs joint | -0.0261 | +0.0000 | 36.5 / 45.0 |
| finance | `public_base` | positionwise staged vs joint | -0.0138 | -0.3906 | 43.0 / 47.5 |
| pile | `public_lora_2601` | past-only staged vs joint | -0.3045 | -0.3906 | 193.5 / 292.5 |
| pile | `public_lora_2601` | positionwise staged vs joint | -0.2107 | +0.0000 | 226.0 / 294.5 |
| finance | `public_lora_2601` | past-only staged vs joint | -0.0538 | -0.9766 | 33.5 / 51.0 |
| finance | `public_lora_2601` | positionwise staged vs joint | -0.0062 | -0.1953 | 42.5 / 44.5 |

## Gate outcome and limitations

The emitted gate fields are `ruled_out_cells={"pile": true, "finance": true}`, `support_cells={"pile": false, "finance": false}`, `harm_cells={"pile": false, "finance": false}`, and `ruled_out_seed_ok={"pile": true, "finance": true}`. The fixed interaction is therefore practically ruled out on this panel, not shown to be mathematically impossible. The transfer target was not used to rescue or overturn that gate.

This is a task-local exploratory comparison of one H128 affine-first schedule, two seeds, two visibility masks, and one 256-record-per-domain panel. Public-fit competence and correction signal qualify the machinery but do not remove finite-sample and architecture limits. The Pile token interval is above zero yet below +0.5 pp; Finance is centered near zero. General staging remains inconclusive. No automatic schedule sweep, confirmation run, sample expansion, or canonical-benchmark replacement follows from this result.

## Reproducibility references

- approved plan SHA-256: `9d1eb8dca89c76f4064c0c380636cb9d585672f7fae796d5295d6507b788caa3`;
- packet SHA-256: `8edf99fcb311eadc1f17b5694dd82001ba6053aefcf8c828a3d0e19a53bb0059`;
- public capture: `experiments/TRR-P08/runtime/public-capture-r3/observations.json`;
- prediction manifest: `experiments/TRR-P08/runtime/predictions-r3/p08_predictions.json` (SHA-256 `f4974384cc760bd304f3ab18b34ea886e554f0bdc171407491647ac19718f8ae`);
- fit receipt: `experiments/TRR-P08/runtime/main-r2/main_fit_receipt.json` (SHA-256 `44b2a965c92d64938e5ff322222678c8cfaf4ba84ff93b62d6f49cfce39d7a86`);
- compact derived score table: `experiments/TRR-P08/review/score-summary-r3.json`;
- tracked truth metadata copy: `experiments/TRR-P08/review/truth-manifest-r3.json`;
- tracked score plan and execution receipt: `experiments/TRR-P08/review/score-plan-r3.json` and `experiments/TRR-P08/review/score-execution-r3.json`;
- full score artifact: `experiments/TRR-P08/runtime/scored-r3/results.json`;
- truth arrays remain outside the repository; no raw truth payload is tracked.

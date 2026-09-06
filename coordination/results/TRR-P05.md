# TRR-P05 result: diagnosing the P04 teacher-ranking regression

**Status:** complete. **Disposition:** ranking imitation improved but was not
useful for reconstruction; retire this tested objective for search-free
inference. No additional P05 teaching experiment is justified by these data.
P04's negative fresh-panel result remains unchanged.

D did learn the cached non-gold teacher order. At the two final D states,
agreement is 68.50% and 68.40% on the 256 difficult rows and 79.17% and
78.86% on the 128 uniform rows. The corresponding affine initial-function
reference is near chance for this pair task (51.39% and 52.03%). D's original
weighted ranking loss falls from 0.8593/0.9029 to 0.4778/0.4802 on difficult
rows and from 0.9029/0.9021 to 0.3173/0.3184 on uniform rows (seeds 1737/2711
where applicable). This is measurable order learning, not a failed transfer
of the ranking signal.

The learned order did not yield a reconstruction advantage. Final D has much
smaller gold-token margins than the same-seed H and S states: 2.19--2.41
logit units lower on difficult rows and 5.19--5.23 on uniform rows. Its
control margins are also 1.48--1.55 lower than the same-seed H/S controls.
Token accuracy is close on this public diagnostic, so the principal quality
gap is margin rather than an across-the-board top-1 collapse. Selected D is
not uniformly improved either: seed 2711's uniform ranking loss is 1.3456,
above the affine reference's 0.9029. The final-state comparison is
explanatory only; it did not alter P04 checkpoint selection.

## Diagnostic and integrity

The diagnostic used only the frozen P04 public correction/replay resources,
cached teacher evidence, the recorded public affine function, and the twelve
stored P04 selected/final S/H/D states. It evaluated 384 teacher positions
(256 `difficult_a1_error`, 128 `uniform_audit`) and 384 same-pool public
controls without teacher scores. It produced 13 forward states and 24
no-update gradient cells. The affine entry is a PR7 public affine
initial-function reference, not a reconstructed initial GRU checkpoint or a
historical trajectory.

For each teacher partition, agreement and ranking loss use the P04 retained
adjacent non-gold pairs and tie policy. Ranking is the exact global weighted
reduction, `sum(weight * softplus) / sum(weight)`, rather than a mean of row
means. There were 7,626 difficult pairs with 60 omitted near ties and 3,817
uniform pairs with 26 omitted near ties. No exact top-1 prediction ties
occurred in the forward sample. The complete forward values are below;
accuracy is shown as correct/rows, margins are gold minus the best other
full-vocabulary logit, and each agreement/loss pair is agreement followed by
the global weighted loss.

| state | difficult top-1 | difficult margin | difficult agreement / loss | uniform top-1 | uniform margin | uniform agreement / loss | control top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine initial function | 0/256 | -1.488181 | 0.513900 / 0.859250 | 115/128 | 5.311015 | 0.520304 / 0.902923 | 354/384 |
| S 1737 selected | 236/256 | 5.442527 | 0.530029 / 1.190781 | 127/128 | 7.716607 | 0.525806 / 1.036331 | 375/384 |
| S 1737 final | 256/256 | 8.021933 | 0.534750 / 1.238508 | 128/128 | 9.474254 | 0.524234 / 1.110425 | 377/384 |
| H 1737 selected | 256/256 | 7.853659 | 0.533438 / 1.229660 | 128/128 | 9.183124 | 0.517160 / 1.085256 | 375/384 |
| H 1737 final | 256/256 | 8.124524 | 0.533045 / 1.238132 | 128/128 | 9.473921 | 0.515064 / 1.099758 | 375/384 |
| D 1737 selected | 256/256 | 5.433066 | 0.608707 / 0.618020 | 124/128 | 3.494381 | 0.656537 / 0.630711 | 372/384 |
| D 1737 final | 256/256 | 5.830708 | 0.685025 / 0.477808 | 128/128 | 4.240380 | 0.791721 / 0.317330 | 373/384 |
| S 2711 selected | 256/256 | 7.700524 | 0.531996 / 1.226184 | 128/128 | 8.742966 | 0.526068 / 1.065750 | 376/384 |
| S 2711 final | 256/256 | 8.304660 | 0.530816 / 1.248649 | 128/128 | 9.406976 | 0.528687 / 1.092489 | 375/384 |
| H 2711 selected | 256/256 | 7.737781 | 0.531865 / 1.226641 | 128/128 | 8.927340 | 0.527640 / 1.073596 | 376/384 |
| H 2711 final | 256/256 | 8.191587 | 0.532520 / 1.244366 | 128/128 | 9.417301 | 0.525544 / 1.092914 | 376/384 |
| D 2711 selected | 253/256 | 4.072368 | 0.548912 / 0.864377 | 106/128 | 2.487159 | 0.545193 / 1.345575 | 369/384 |
| D 2711 final | 256/256 | 5.891951 | 0.683976 / 0.480214 | 128/128 | 4.212885 | 0.788577 / 0.318399 | 375/384 |

The final D totals are 757/768 and 759/768 for seeds 1737 and 2711. The
same-seed final H totals are 759/768 and 760/768; final S totals are 761/768
and 759/768. Controls have no teacher-ranking metrics and were used only for
public reconstruction comparison.

## Objective interaction

The gradient cells preserved the original P04 reductions: full CE is a mean
over active rows, hard confusion is a mean of per-row negative means, and
ranking is the global pair-weighted softplus mean with coefficient 0.25. The
reported ranking-to-CE ratios below use the weighted ranking gradient norm,
not scalar losses or the coefficient alone. The cosine is with the gradient
of negative gold margin, so a positive value means ranking descent points
locally toward a larger gold margin.

| stored checkpoint | active cells | weighted ranking norm / CE norm | rank cosine with negative gold margin | rank cosine with CE | clip factor |
|---|---:|---:|---:|---:|---:|
| selected H | 4 | 60.6--140.4 | -0.2629 to -0.0469 (4/4 negative) | -0.2222 to -0.0601 (4/4 negative) | 1.0 in all cells |
| selected D | 4 | 0.7982--6.7198 | -0.3364 to +0.0290 (1 positive, 3 negative) | -0.3245 to +0.0527 (2 positive, 2 negative) | 1.0 in all cells |
| final D | 4 | 1.0607--2.4110 | +0.2071 to +0.7344 (4/4 positive) | +0.1896 to +0.6989 (4/4 positive) | 1.0 in all cells |

The selected-H counterfactual rank gradient is therefore large relative to
CE and locally opposes gold-margin improvement. At selected D its direction
is mixed, while at final D it is aligned with both CE and the margin
component. This is a local diagnostic at stored states, not evidence of the
historical parameter updates: optimizer moments and the full intermediate
trajectory were not available, no parameter update was called, and every
state digest was unchanged. The mixed D result and positive final-D alignment
do not support attributing the P04 regression to one reproducible scaling
conflict.

The probe is intentionally sparse. The eight unique original P04 schedule
batches contain 3,987 selected positions, but only four batches contain
teacher rows: steps 999 and 1999 for seed 1737, and steps 0 and 1999 for seed
2711. Those batches contain only eight unique teacher rows, all
`uniform_audit`, and 239 retained pairs. Reusing each batch at three states
makes 24 teacher-row and 717 pair observations across cells; it does not make
24 independent teacher batches. The gradient results cannot diagnose
conflict on difficult teacher rows and should not be generalized to the
whole training trajectory. Zero-row cells have no ranking gradient or rank
cosine; they are not zero-cosine evidence.

## Decision and limits

This meets the first packet disposition: the fixed P04 ranking objective
learned its additional teacher-order signal, but that signal did not improve
the reconstruction objective in the stored-state comparison. D's lower
margins against S/H, together with P04's already negative fresh-panel result,
make the tested ranking objective unsuitable for search-free inference. The
local gradient evidence is useful for explaining why a conflict is possible,
but its sparse uniform-only coverage and state dependence are insufficient to
claim a historical optimizer or coefficient mechanism. No follow-up teaching
run is recommended in this task.

This is a conclusion about the tested P04 D objective and its fixed training
setup. It is not a claim that every teacher-ranking formulation or every
teacher signal is ineffective. P03 remains stopped; no P03 holdout, agent-one
artifact, target data, or hidden truth was opened.

## Reproducible evidence

The full run receipt reports `PASS` at source commit
`e022b56ff92b4987b88a47418b4df76ccd296cea`, with no truth access, no
optimizer step, 13 forward states, 24 gradient cells, and wall time
40.60715087399876 seconds. The resource guard recorded maximum host RSS
5,961,519,104 bytes, maximum recorded CUDA reserved memory 2,998,927,360
bytes, and minimum recorded free CUDA memory 12,590,252,032 bytes, within its
16 GiB host, 6 GiB reserved, and 8 GiB free limits. The watchdog also passed.
The focused implementation validation was 7/7 tests at the same source
commit; the compact derivation script was rerun from the immutable summaries
and gradient receipts.

The exact diagnostic command and all input hashes are recorded in the run
receipt and watchdog command receipt. The main evidence is:

- `experiments/TRR-P05/runtime/diagnostic-r1/diagnostic_receipt.json`
  (SHA-256 `07f55da1fc02fca9f2b7a65b5875f0159a54dd9171374f22896ccbda14ca6b59`)
- `experiments/TRR-P05/review/diagnostic-derived.json` (generated by the
  review script from the receipt and compact forward/gradient summaries)
- `experiments/TRR-P05/review/summarize_diagnostic.py`

The diagnostic output is task-local, public-only, and retained with the
P05 runtime evidence; no truth payload is part of this report.

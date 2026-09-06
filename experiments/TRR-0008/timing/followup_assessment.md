# TRR-0008 alias-control precision assessment

This note analyzes the completed initial diagnostic without changing its result,
thresholds, or order schedule. The source receipt is
`experiments/TRR-0008/timing/result.json`, SHA-256
`215f5bb3c1597e6970bd84fc55f9990b7250e6f29f54a100d2cded660a9b9c53`, produced
by code commit `74deda63e7fcf8ff5bf7e563adc5089fd07189e4`.

The exact archived-prediction equivalence check passed for all five methods and
all four cells. The identical-class and identical-tensor alias check passed.
The alias runtime control is nevertheless `INCONCLUSIVE`, because the fixed
[0.95, 1.05] containment rule is not met in two cells while neither interval
lies entirely outside the band:

<!-- Generated from result.json with this receipt parser; JSON is authoritative. -->
| Cell | Ten alias/reference block ratios | Mean | 95% CI | Control status |
| --- | --- | ---: | --- | --- |
| `finance__public_base` | 1.060845864624591, 0.9961118824580061, 1.0042991553623537, 1.0373422674884196, 1.0260030877454427, 0.9052073248354929, 1.0175247464034751, 0.9636528387069496, 0.9708948284042941, 0.9834877531346363 | 0.9965369749163662 | [0.9650647496041413, 1.028009200228591] | contained |
| `finance__public_lora_2601` | 0.9231854189243769, 0.8955398125977422, 0.9534095104509235, 0.9884510975764529, 1.0004624034765002, 1.035031389154572, 1.0012266208674325, 1.074933389502083, 0.9577051492517843, 0.9954484857037155 | 0.9825393277505583 | [0.9450475193089013, 1.0200311361922154] | inconclusive overlap |
| `pile__public_base` | 1.0008951122748977, 1.011160612859708, 0.9907014544061425, 0.8948077229118189, 0.9911368744580142, 0.9622693916003194, 0.9701228869240883, 1.0097914259813041, 1.0116849693877032, 1.0645044009777196 | 0.9907074851781715 | [0.9594331642036272, 1.0219818061527157] | contained |
| `pile__public_lora_2601` | 1.0977330314409224, 1.0670347480195308, 1.0630501075924679, 1.0818541032099473, 1.028322494969012, 0.9671729777667482, 0.9611789652350813, 1.0136663705377902, 0.9650826733326665, 1.1231917259789395 | 1.0368287198083106 | [0.9947594162260333, 1.078898023390588] | inconclusive overlap |

The first five rotation blocks versus the five reversal blocks have descriptive
mean ratios of:

| Cell | Rotation blocks 0--4 | Reversal blocks 5--9 | Reversal minus rotation |
| --- | ---: | ---: | ---: |
| `finance__public_base` | 1.0249204515357626 | 0.9681534982969696 | -0.056766953238793015 |
| `finance__public_lora_2601` | 0.9522096486051991 | 1.0128690068959174 | +0.060659358290718314 |
| `pile__public_base` | 0.9777403553821162 | 1.0036746149742268 | +0.025934259592110688 |
| `pile__public_lora_2601` | 1.0675988970463761 | 1.006058542570245 | -0.06154035447613095 |

These paired shifts are large relative to the row-level noise and have mixed
signs across cells. The alias/reference block-ratio SDs are 0.0439951,
0.0524099, 0.0437185, and 0.0588088 for the four cells in the table order.
Using the measured within-block row CVs, the independent row-noise projection
for a 32-row sum is only about 0.012--0.016. Thus adding rows alone is
unlikely to resolve the control: after subtracting that component in
quadrature, the residual block-scale SD remains approximately 0.042--0.057.
The block walls were stable (7.446--7.665 s), and resource/thermal guards did
not show an anomaly; the mixed half-to-half shifts are evidence of order/phase
sensitivity, not a basis for selecting a faster order.

## One bounded follow-up option

If root elects to spend one further timing window, pre-register a **40-block
total** run: four complete repetitions of the same ten-block design (five
fixed cyclic rotations followed by their reversals), all four cells, the same
32 rows per cell, all five methods, one warmup, the same synchronization and
timed boundary, and the unchanged alias/candidate CI rules. The four cycles
must be specified before execution and run in their fixed order; no order is
selected from the initial durations. Write a new create-only follow-up result
and leave the initial receipt immutable.

Forty blocks should fit the existing 600-second guard with margin: the initial
10-block run took 75.48 s for block execution and 89.10 s overall, so a
four-cycle estimate is roughly 315--320 s before ordinary run-to-run variation.
Under the observed means and SDs, a 40-block Student-t projection gives
`finance__public_lora_2601` approximately [0.9658, 0.9993] and
`pile__public_lora_2601` approximately [1.0180, 1.0556]. It should therefore
resolve the finance-LoRA containment question and materially narrow the
pile-LoRA interval, but it does not guarantee alias PASS for pile-LoRA. A
further opportunistic extension should not be planned in advance: the one
follow-up either resolves the fixed criterion or leaves the alias control
inconclusive.

Increasing rows to 64 or 128 while retaining ten blocks is less efficient for
this receipt. It reduces the already-small row-noise component but leaves the
observed block/order component largely unchanged. No threshold or decision
rule is changed by this assessment.

The communication-scope receipt is in
`experiments/TRR-0008/timing/status.json` under `unpublished_parallel_access`:
P06 was read once while locating the parent, scheduling/coordination status only
was encountered, and no unpublished scientific findings were used. The same status file records the harness test history: the committed baseline
was `8 passed`, the post-regression harness was `9 passed`, the current harness
with the 40-block schedule guard is `12 passed`, and the df=39 Student-t
regression uses `scipy.stats.t.ppf(0.975, 39) = 2.022690920036761`; the
compatibility count is `29 passed`. The earlier committed `8 passed` field was
stale before those final CPU checks.

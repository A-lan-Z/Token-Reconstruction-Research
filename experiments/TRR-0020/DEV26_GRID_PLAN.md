# Development26 reconstruction: full causal scaled inverse

Public conditioning qualification completed48cases and144repetitions. All24 archived identity controls
are exact. The coordinate-probe rule reduces both linear and nonlinear mismatch versus the unscaled
global solve on both public lengths. At16steps/ridge1e-4 the128-position nonlinear half-step ratio
falls from.624 to.573 (about.151s including scale preparation versus.135s); at40 it falls from.591
to.560. These are numerical comparisons, not reconstruction scores. Even32steps remains above the
older diagonal half-step mismatch, so this grid explicitly tests repeated updates and full steps.

Freeze18 configurations before execution: warm0/32/64 full-vocabulary Gini003 steps;
8/16/32 global CGLS iterations; nonlinear damping.5/1. Ridge multiplier stays1e-4 and scaling
is coordinate_probe4 with the exact qualified constants. Each configuration makes16 non-linear
updates and emits all four existing full-vocabulary readouts at updates4/8/16.
Eight fixed retrospective development records yield144 reconstruction cells and1728
checkpoint/readout variants. Largest warm64/CG32/damping1 runs first in an isolated process.

The initial full-vocabulary soft optimizer is unchanged. All128256 vocabulary entries remain
available. There is no token shortlist, no intermediate token filtering and no separate token
verifier. BOS remains fixed. Prefix weights remain frozen; optimization changes current input only.
Initialization uses the supplied prefix-derived metric, so this is not lookup-free.

Use the qualified full causal J/JT and global sequence inner products. At each nonlinear step,
four deterministic Rademacher output probes estimate current normalized Jacobian-column sensitivity.
Their seed is200063+native length; they are fixed across steps/replays. Equal-weight coordinate/position
shrinkage, the1e-6 median-sensitivity floor and median1 scale normalization are unchanged.
CGLS regularizes scaled coordinates. Limit each position's physical update norm to the public
median vocabulary embedding norm, then apply the configured damping.

Required gates:
- each configuration: three replay executions and one uncaptured local-stage execution at128/40
  positions on the same public seed200051 fixture, with exact output and warm-trace equality;
- independent CPU float64 readout formulas and a global scaled-J/JT duality check at both final
  public states; no invalid per-position duality claim for this coupled operator;
-144 archived Gini warm anchors and96 archived initial-mixture anchors;
- native shapes, finite artifacts, fixed BOS and zero solver-error diagnostics;
- complete source/asset/output freeze before retrospective labels; negative missing-cell,
  changed-source and changed-output truth-gate tests.

Count warm-stage prefix forwards/backwards separately from17 continuous forwards,16*K analytic
JVPs,16*(K+5) analytic VJPs (including the four scale probes),6 full-vocabulary readout products
and one mixture product. Record model/table preparation, random-probe preparation, CUDA capture,
input optimization, synchronization, output transfer and storage. Earlier checkpoint times include
earlier diagnostic readouts; these development timings are not a paired canonical speed comparison.

No new active canonical method is registered from this prospective grid alone. If a fixed rule is
adequate, register it and run both canonical setups with the complete active matrix. All panels
here are historically opened, supplied-public-prefix evidence, not fresh or recovering-prefix evidence.

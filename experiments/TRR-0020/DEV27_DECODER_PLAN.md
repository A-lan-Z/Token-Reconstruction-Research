# Development27 committed-context decoder: frozen execution plan

Implement the already declared eight-configuration family from DEV27_NEXT_DECODER_PLAN.md.
Configuration order: warm0/64, polyak0.25/cg1, updates1/4. Run the largest warm64/cg1/4
configuration first, including three repeated public128 and public40 whole decodes.
Every configuration receives the same public checks and all eight retrospective inputs.
There are64 reconstruction cells. None is an active canonical method at this stage.

Initialization is the unchanged Gini003 full-vocabulary soft mixture, with0 or64 optimization
steps. At each unknown position, use only the BOS and previously emitted IDs to build the
immutable past K/V. Apply exactly1 or4 normalized current-token updates using the qualified
analytic map and direction, clipped to the median token embedding norm. Choose exactly once
by the existing white-cosine argmax over all128256 rows. The implementation computes the
existing four score formulas (two vocabulary products); charge both products. No K-list or
discrete verification exists. A selected token is forwarded solely to commit subsequent K/V.
Skip the final token's commit because there is no subsequent position. Total commit forwards
equal T-1, including BOS. Prefix weights remain frozen.

New numerical identity: uncaptured FP32 current-token computation, TF32 disabled, variable
native cache length, with unchanged BF16 warm optimizer. Do not claim native BF16 equivalence.
All per-stage CPU/GPU synchronization is included in elapsed inference; derivative sub-times
overlap the direction time and are explicitly labelled. Preparation, warm capture and disk I/O
are reported separately. The total includes all diagnostics and unused readout formulas.

Controls: three exact repeated outputs and warm traces on public128/40, independent float64
readout formula checks, and independent full-sequence causal reference at positions1/middle/last
using emitted IDs and the saved initial continuous current row. Cached-vs-full causal reference
allows rtol5e-4, atol1e-4 because GEMM geometry differs; this is numerical qualification, not
exact-output equivalence. Recomputing the same single-row readout must reproduce the emitted ID.
Validate cache lengths and committed IDs. Require all64 exact warm anchors to dev7 and all64
exact initial-mixture anchors to dev26 (warm0 andwarm64). Preserve failed outputs before gates.
Complete64-cell freeze, source/asset hashes, and negative missing/source/output gates precede
any current read of retrospective labels. Archive raw artifacts with byte hashes.

Source truth and synthetic public IDs are never passed to decoder; only Hcut is passed.
Public IDs generate the synthetic observation and are not used to assess reconstruction quality.

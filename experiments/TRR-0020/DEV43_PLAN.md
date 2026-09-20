# Development43: complete fused scalar-update reconstruction

Compare the unchanged cold factor2/four-scalar-iteration optimizer with the
qualified tiled scalar variant, each independently stopped at64and128updates.
Use the existing eight development observations: two each matched/LoRA256 x
prose/stress, native128/40lengths. Four exploratory rules x eight inputs gives
32cells; repeat everycell three times. No truth enters generation, stopping,
selection, timing or update. Freeze all32cells before retrospective scoring.

The only numerical change in fused mode is development42's scalar evaluation.
Keep full128256vocabulary support, original initialization and readouts, known
BOS, prefix computation, probability gradient, factor2, four scalar iterations,
and all update constants. No response lookup, candidates, verifier or model
fitting is added. Retain the native geometry and explicit preparation costs.

Execute isolated workers for fused and control, sequentially. Qualify largest
128length/128updates first perworker. On public seed200051at128/40lengths and
64/128updates, require three exact repeated decodes, one complete eager/replay
comparison percell, and an independent probability-gradient check perlength.
All warm/initial anchors must equal the archived development38control.
Control64full outputs and control128's64-step prefix/trace must match archived
control64exactly. Fused later outputs may differ and are evaluated as a new
numerical variant. This gives24public repeated decodes,eighteager checks and
four independent gradient checks acrossbothworkers.

For real inputs require exact three-repeat arrays and traces, all unchanged
initial anchors, and control64/prefix anchors. Save phase timings, work counts,
memory, raw hashes, source binding and environment. Restart only completed
hash-verified cells; preserve partial failures. The scorer verifies allcells,
public qualifications and raw hashes, and rejects missing-cell, changed-source
and changed-output negative fixtures before reading retrospective truth.

Report every final/best-objective/best-position readout, bothconditions and
groups, without pooling or claiming canonical replacement. The four rules are
an exploratory grid, not registered active methods. No winner is selected using
truth during generation. If a contender is advanced, register its frozen rule
and complete both canonical setups with fresh paired controls and full matrix.

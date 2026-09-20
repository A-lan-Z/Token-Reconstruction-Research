# Development45: full-decoder exact pointwise execution
Preregistered exploratory execution test; no new active canonical method.
Integrate development44 pointwise kernels without changing original native
Torch reductions, FP32 arithmetic order, scalar iterations (4), budget factor
(2), initialization, stopping budgets, full vocabulary, or output readouts.
No A1, shortlist, separate candidate verifier, or fitted model is used.

Two implementations (original control, exact pointwise) at actual 64 and128
updates on the existing eight development observations form32cells,3repetitions
each. Both implementations must reproduce every archived development43
original-control output tensor byte for byte and all recorded warm/mirror
traces, at the matching actual budget. Signed zeros count as differences.
Any disagreement aborts and preserves the attempt unscored.

Before each worker releases its development matrix, use public seed200051,
native lengths128and40, both stopping budgets, three replay repetitions and
one eager check per length/budget. Start with length128/budget128. Check
independent probability-gradient products. This is execution qualification on
recorded outputs and traces, not an exhaustive proof over all possible states.
No padding, microbatching, or altered numerical geometry.

Run modes in isolated sequential processes, exact then control (not randomized
between modes). Freeze all32cells before retrospective scoring. The scorer
checks complete coverage, bindings, public qualification, work counts, hashes,
and three negative controls before opening evaluator truth. Existing public
controls, source and observation hashes are bound. Keep all readouts, including
ones not chosen by token correctness. Publish native end-to-end timings
including initialization, updates, diagnostics, synchronization and transfers;
report prefix/engine preparation, graph captures and disk I/O separately.

This is the supplied public cut4 Llama3.2-1B prefix, not an actual recovering
prefix training trajectory. Existing development truth has already been opened
in earlier tasks. Do not claim fresh confirmation, new canonical comparison,
or baseline replacement. The68cell active canonical matrix remains unchanged.

Resource preflight: prior complete decoder peak4.201GiB; isolated exact kernel
peak2.635GiB, original fixture preparation4.030GiB. Conservative full decoder
estimate6.5GiB reserved with max127x128256 logits and gradients, native
128x2048 prefix geometry. Require at least10GiB free GPU before launch; runtime
guard keeps2GiB GPUfree and8GiB hostfree, RSS<=10GiB,temp<80C, timeout900s.
Inner guard retains3GiB free and8GiB reservation ceiling. Estimated runtime
<=300s from prior183.706s32cell run and modestly lower kernel throughput.
No overlapping GPU work. Qualify the largest cell before the remaining matrix.

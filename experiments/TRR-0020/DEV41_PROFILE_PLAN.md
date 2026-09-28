# Development41: current full-vocabulary optimizer execution profile

The current cold factor2 / four-scalar-iteration optimizer is substantially
different from the old development5profile. It retains every vocabulary entry,
passes only a continuous probability mixture through the prefix, and does not
generate a shortlist or run a separate candidate verifier.

Use the unchanged public seed200051fixture and native128/40geometries from
development38control. Reproduce its64-step outputs and complete traces exactly
three times pergeometry, including the archived anchor. Largest128runs first.
Record fresh full inference timings and preparation/capture separately; these
are public execution checks, not benchmark accuracy or an A1comparison.

For eachgeometry independently reset to the16-step state. Profile five
unchanged full optimizer updates with CPU/CUDA operator shapes. Reset again
to the same16-step state, compute its probability gradient once, then profile
five calls to the unchanged KL-budget update without writing back logits.
Check those repeated update arrays exactly and verify inputs are unchanged.
Save profiler traces and shape-grouped operator tables. Compare device work
and wall time, with explicit warning that profiling perturbs latency and the
isolated update uses a fixed state rather than an evolving trajectory.

No method decision rule changes in this run. A kernel implementation, if the
profile justifies one, requires separate numerical qualification. Float
association changes will not be silently called an exact execution port.
Native sequence/vocabulary geometry is unchanged.

Preserve full logits, probability gradient, observed-error budget inputs and original outputs at the16-step state as public numerical fixtures for any subsequent kernel qualification.

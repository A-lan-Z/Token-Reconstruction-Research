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

## Revision1: release the profiling-only autograd reference

The original process completed all128-position work, then failed while capturing
the40-position graph with cudaErrorStreamCaptureImplicit. The local loss tensor
from the isolated gradient evaluation remained alive across geometries and kept
its default-stream AccumulateGrad reference. The process is terminal; the failure
and all partial artifacts are preserved, excluded from the complete profile.

Release that local loss immediately after backward. This does not change the
decoder, gradients, sequence geometry, update rule or candidate handling.
Run the entire same matrix again in a fresh isolated process. Require all public
decoder anchors and repeated updates exactly. No driver/allocator memory anomaly
was observed; the cause is an explicit stream dependency during graph capture.
Retain the original source at its execution commit and the partial archive.

GPU totals are computed directly from raw kernel/memcpy/memset trace events. Summing both CPU operator attribution and kernel self-times would double count work; original partial totals are excluded. Keep the complete operator tables for attribution only.

## Revision2: detached fixtures and isolated geometry workers

Revision1 again failed at40-position capture despite releasing loss. The
profiling harness also retained a differentiable logits view across geometries.
Its entire128-position partial output and corrected event traces are preserved;
the attempt is excluded from the complete matrix.

Detach the read-only logits/gradient/error fixtures. More generally, execute
each native geometry in a separate sequential process to remove profiling
autograd/stream state from subsequent graph capture. This changes preparation
isolation only; it does not batch, pad, resize or alter the decoder. Count both
processes' preparation separately. Require every64-step public output and full
trace to equal the same archived controls. This is an output-equivalence check
for the isolation change, not an assumption of neutrality. Largest128worker
must pass all profile and resource checks before launching40.

GPU stream-capture errors were terminal and preserved. Neither attempt reported
an allocator, thermal or hardware-driver anomaly. Fresh processes avoid reusing
the invalidated capture context. There is no live process from either failure.

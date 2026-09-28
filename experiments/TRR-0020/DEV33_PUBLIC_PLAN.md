# Development33 actual-prefix public qualification

The prior turn advanced the goal:56reconstruction cells,60public cases and the
causal probability-chain CPU primitive completed. Save continuation16 before edits.
Older scope check confirms the relevant sequential decoder (development27) adjusts
a continuous embedding after a joint warm start; development9 uses parallel hard
context. Neither is this current-token full-vocabulary probability optimization.

Freeze72public cells: T128/40 x current positions(last,middle,1) x probability-budget
factor2/1/.5 x updates8/4/2/1. Three repetitions each. Use immutable known synthetic
past only to qualify the mechanism. It is not hidden-input reconstruction.

Use the qualified FP32 current-position prefix map and VJP, native context lengths,
TF32 disabled. Every vocabulary token remains eligible. Initialize logits as80times
the current observed activation's normalized prefix-metric similarity to every token
embedding. The mixture is FP32 p@E. Its probability gradient is the qualified current
input VJP times E transpose. This is a new FP32 rule, not a byte-equivalent port of
the prior joint BF16-mixture method. Use the already-qualified four-iteration
observed-error KL update with budget clamp(factor*cosine_error,0,1).
After the requested fixed updates, emit full-vocabulary logit argmax. One additional
current-prefix forward commits that emitted token's actual embedding for history;
this commit does not score or select alternative token candidates.

Before the matrix, require all six unchanged development27 noisy-input forwards
and raw JVP/VJP anchors. At every public context compare the new full128256-coordinate
probability gradient to independent autograd through the HuggingFace prefix with
fixed past. Existing FP32 reference tolerances:rtol5e-4/atol5e-5. Preserve all
reference arrays, logits, mixtures, outputs, scalar traces, actual FP64 KL at every
update, and emitted-token commit outputs/KV. All repeated arrays must be exact.
Budget feasibility tolerance2e-5 is unchanged. Preserve failures and stop on anomalies.

Measure preparation, full-vocabulary initialization, synthetic-history generation,
each update's mixture, forward, cosine cotangent, VJP, vocabulary-gradient product,
probability update, and final emitted-token commit. CUDA events separate update
components; synchronized wall time measures the eager trajectory. Diagnostic logits
clones are included; FP64 KL validation/serialization are separate. This is not yet
optimized or a decoder latency claim.

Run the largest native position first, with the external exclusive GPU resource
guard and internal8GiB reserved,3GiB GPU free,8GiB host available margins.
No new active reconstruction method/canonical cell is registered at this numerical
stage. If useful, next qualify exact CUDA replay and the entire native128-position
cache/capture geometry, then preregister a causal reconstruction grid. No numerical
padding or batching change is assumed neutral.

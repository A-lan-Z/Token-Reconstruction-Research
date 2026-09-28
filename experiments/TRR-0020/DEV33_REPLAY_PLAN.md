# Development33 exact replay and largest native geometry qualification

The72-case eager public study completed216repetitions and all primitive/reference
checks. Factor2 reduced the observed error across all six contexts substantially;
four/eight updates were strongest. No hidden reconstruction accuracy is known.
Eager execution is costly: about9ms/update, with mixture/gradient products taking
about2.47ms combined and the prefix forward/VJP about4.15ms combined.
Test execution improvement without changing this rule.

Implement static buffers for the single current observation, full128256logits,
eight-step diagnostic history, fixed-size history storage, rotary vectors and
factor. Captures use each native current position0..127; the attention operation
receives only history[:position], not padded attention. The buffers store up to128
positions, but no numerical padding/batching is introduced. Factor is a GPU scalar;
its equivalence to the old Python factors.5/1/2 must pass all public controls.

Qualify the full128-position graph/cache footprint first. Independent capture pools
and one owned capture stream; no speculative shared-pool reuse. Every capture reads
the same weight storage. Internal resource guard8GiBreserved/3GiBfree/8GiBhost.
Estimate graph transient memory before launch and stop/reassess on unexpected growth.
The largest context is captured first, then the complete set of native positions.
Source/input/model hashes and preparation/capture/peak memory must be recorded.

Compare replay with every one of the72previous public cells and three repeated runs
per cell. Save all same arrays and require exact equality, including full logits,
soft outputs, per-step traces, direct-FP64 KL, and emitted-token commit/KV. Also run
an eager execution of the new buffer adapter for every cell and require equality.
Changing geometry or approximation is not allowed to repair a failed equality gate;
preserve the attempt and investigate. Prefix-history storage before the current
position must remain unchanged.

Replay timing includes full-vocabulary initialization, unlike the old public
trajectory timer, which begins with initialized logits. Do not subtract or compare
these as identical end-to-end timing boundaries. Initialization, commit and
diagnostic transfers must be disclosed; no decoder speed claim yet.
After qualification, define the actual causal decoder and require full-sequence
repeatability before freezing a retrospective reconstruction matrix.

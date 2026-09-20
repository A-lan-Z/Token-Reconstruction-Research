# Development44 GPU qualification details

The unchanged CPU-qualified definition remains in DEV44_PLAN.md.
## Frozen fixture and GPU matrix details

Generate original public states after0,64and128updates at each native length
128and40, seed200051. Each length has an isolated preparation process. Compare
the0-step initial/warm outputs and64/128full outputs/traces to archived original
development43public controls. Freeze logits, probability gradient, observed
errors and original expected update. No benchmark observation or label is used.

The GPU matrix has20cases:12tiny, two existing step16fixtures and six newly
frozen states. Trace every primitive against its torch expression, then require
three byte-identical full updates and input immutability. Compare both eager and
GPU-replay arrays. Time all eight full-vocabulary states in three rotating groups
of20calls per implementation:960timed graph calls. Primitive signed-zero and
subnormal cases are additional checks, not reconstruction evaluations.

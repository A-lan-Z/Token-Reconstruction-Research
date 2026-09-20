# Development46: exact reuse of forward probabilities
Exploratory execution qualification; no new active canonical method.
Avoid the optimizer's duplicate native softmax by retaining the detached
probability already produced by PreconditionedSoftmax in the same forward pass.
No logits update occurs between that forward and use. Keep the original
log_softmax, reduction operations, constants,128256 vocabulary coordinates,
BF16 mixture/probability-gradient products and FP32 prefix.

CPU:24 independent original-update comparisons across FP32/64, vocabulary
17/1023/1031, random/peaked/flat/zero-budget cases; reuse input must not mutate.
GPU:20 cases using eight bound original public states0/16/64/128 at native
lengths128 and40 plus12 tiny cases. Compare complete outputs and every returned
diagnostic byte for byte, check unmodified cached probability and inputs,
repeat three times. Eight full-state isolated paired graph benchmarks rotate
order across three groups of20 calls. This excludes the reusable probability
calculation from the update cost because it is already required by the forward
pass; full decoder timings are the governing speed evidence.

After the largest public update qualifies, run complete original-exact versus
reused-probability decoders on the existing eight development observations,
at actual64/128 updates, three repetitions each (32cells). Each mode first
qualifies largest128-position128-step public replay/eager execution, then both
geometries/budgets, and independent probability-gradient products. The reuse worker also compares cached probability,
independently calculated embedding-cotangent gradients, original frozen gradients
and errors at all eight public0/16/64/128 states, retaining full probability/
gradient/error arrays. Every output
tensor and all recorded traces must match the archived development45 exact
implementation byte for byte. Detached probability references avoid retaining
an autograd history across capture streams. No resource geometry changes.

Separate sequential mode processes, control then reuse. Not randomized mode
order. Preserve failures, never silently restart or alter numerical geometry.
Bind all sources and observations. Freeze the complete32cell matrix before
retrospective truth scoring and enforce three negative truth gates.
Report input initialization, optimization, diagnostics, synchronization and
output transfer; prefix/engine preparation, capture and artifact I/O separately.
Keep all declared readouts. No candidate proposal or model fitting is introduced.

The panel uses supplied public prefix weights, not a recovering-prefix
trajectory. Existing truth is retrospective. This execution experiment cannot
fix earlier reconstruction errors by itself or establish baseline replacement.
The complete34-method68-cell canonical matrix remains unchanged.

Preflight: prior full decoder reserved peak4.576GiB; estimate7GiB including
held probability arrays and graph pools. Largest probability127x128256FP32
is62.14MiB; retaining two native geometries costs<82MiB per live set.
Require>=10GiB free GPU, no other compute processes, host>=8GiBfree,
retain>=2GiB GPUfree at runtime, inner guard>=3GiBfree/reserved<=8GiB,
RSS<=10GiB,temp<80C,timeout900s. Expected total<=350s including kernel checks
and32cell decoder matrix based on prior167.64s decoder plus42.15s qualification.

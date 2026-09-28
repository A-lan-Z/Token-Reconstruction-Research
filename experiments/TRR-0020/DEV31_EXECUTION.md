# Development31: exact probability-change calibration, public numerical diagnostic

The preceding clarification turn added no experiment evidence. Resume the previously
committed development31 plan. No method has met the complete no-shortlist objective.

For each row normalize the probability gradient to d=(G-min G)/(max G-min G).
This common shift/scale preserves the exponential path. Set h=d-E_p d,
c=E_p h, and a=logsoftmax(z). In exact arithmetic
K(t)=logsumexp(a-t h)-logsumexp(a)+t c.
The derivative is c-E_q h and its derivative is Var_q(h)>=0, so K is nondecreasing
from zero for t>=0. A minimum h coordinate j gives
K(t)>=a_j-logsumexp(a)+t(c-h_j).
Thus (target-a_j+logsumexp(a))/(c-h_j) brackets the target when the denominator
is positive. The minimum coordinate bounds only one scalar; no token is proposed.

Use 16 safeguarded Newton iterations, each with two full-vocabulary reductions
(logsumexp and softmax expectation). Keep lower/upper brackets. Newton proposals
are multiplied by .999 to encourage a feasible lower iterate; outside proposals
become bisections. Return only the tested feasible lower bracket. Target is
tau*(1-1e-4). The inactive numerical case is constant gradient or an underflowed
zero expected gap to the minimum. No vocabulary entry is masked or truncated.
A finite-arithmetic overflow cap on the scalar upper bound is disclosed and the
actual bracket must pass qualification. This is forward-KL calibration of an
exponential mirror path, not an exact forward-KL constrained optimizer.

CPU qualification independently bisects direct FP64 KL for 160 steps, tests
monotonicity, brackets, finite outputs, probability agreement, FP32 rounding,
zero gradients, tiny gradient scales, and extreme finite logits. All failed
attempts are preserved and block GPU execution. Root-relative tolerance .004,
FP32 budget tolerance2e-5; FP64 budget tolerance2e-10. Inactive underflow rows
are disclosed and not compared to an unattainable finite-precision budget.

Conditional GPU public study remains24cases, three repeats: T128/40 x warm0/64
x tau.01/.1/1 x span4/KL. Same fixed public seed200051 and original warm-rule
control. Compute a single BF16 surrogate probability gradient through the full
prefix, then apply each update independently. Evaluate actual nonlinear observed
cosine loss, full-distribution FP64 KL, confidence, and wall time. No input labels
are opened; synthetic public token IDs are used solely to generate the fixture.
The study is not a token-accuracy benchmark or an active canonical method.

Record full source, environment, assets, CPU result and preflight bindings. Save
public fixture, baseline mixture/output/gradient samples, updated mixture/output,
token readout, confidence/error and scalar diagnostics. Check full logits byte
equality across repeats in memory, plus exact old-control reproduction.
Graph/forward numerical adaptation requires an explicit equivalence check.

Preflight precedes GPU execution: estimate simultaneous vocabulary arrays and
compare live resources. Largest128x128256 geometry first, no concurrent GPU jobs;
fail-closed external guard and internal8GiB reserved/3GiBfree/8GiBhost margins.

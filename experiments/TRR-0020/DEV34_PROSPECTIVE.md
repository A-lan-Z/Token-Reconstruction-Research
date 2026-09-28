# Development34 prospective diagnostic

Status: not implemented, no new runs claimed.

Development33's96-cell causal decoder is too slow and inaccurate. Its public
step-size audit shows that the four-iteration scalar solve already uses at least
99.758% of every requested budget in the eighteen full trajectories. Increasing
root iterations alone has little numerical headroom on these fixtures.

The next concrete action is a CPU-only descriptive audit of the already frozen
decoder outputs. Bind the complete freeze, scored truth file and output hashes
before reading labels. For each of the twelve configurations and eight records,
report the first incorrect token's position, final mixture error and confidence,
then the same values for later errors. Keep matched/shifted and prose/stress
groups separate. Do not decode or publish source plaintext, change a submitted
output, select an oracle checkpoint, or use correctness as an inference signal.
This retrospective analysis may motivate a new method but cannot establish its
generalization.

Use that audit to separate questions of insufficient current-token convergence
from propagation after an earlier emitted error. Neither cause is yet proven.
If low-confidence, high-error first mistakes dominate, a public-only test of a
larger observed-error budget/cap is a concrete next optimizer experiment; treat
every changed fixed constant as a new decision rule and retain the factor2/cap1
control. If first mistakes are confidently wrong despite low mixture error,
additional iterations or faster kernels alone do not address the readout gap.

A lower-precision current-token backend remains a separate possible speed
experiment. It must be labeled a new numerical method, with independent
forward/gradient references and its own complete matrix; it is not an exact port
of the current FP32 method. No precision, attention padding, batching or graph
pool change may silently replace the qualified implementation.

Before a new GPU grid, perform a measured resource preflight and public qualification. All vocabulary coordinates remain eligible;
do not restore a shortlist or a separate candidate verifier. Register any method
selected as an active contender before its full two-benchmark evaluation.

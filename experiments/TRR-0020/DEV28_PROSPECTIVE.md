# Next public diagnostic: current-token metric preconditioning

Development27's64-cell committed-context decoder is complete and rejected. With warm64,
four cheap updates gain only one shifted-prose token (237to238/254) and improve no other
group. All warm64 directions were below the0.93326 embedding-norm cap: the failure is not
caused by clipping. Continuous cosine error decreases across the four iterations while
discrete readout usually remains unchanged. This does not establish impossibility.

Before another reconstruction matrix, test whether an inexpensive fixed input metric can
improve the qualified current-token direction. Use the existing public noisy fixtures and
six immutable synthetic contexts from development27; these are algebra/nonlinear-response
diagnostics, not reconstruction accuracy.

Use exactly three SPD preconditioners: identity, normalized A, and normalized inverse A,
where A=R R^T + 1e-3 trace(R R^T)/D I and R is the existing prefix-derived readout transform.
Normalize each nonidentity operator by the median of its diagonal. Record the complete
definition, eigenvalue/condition diagnostics, construction time and memory. No fitted
parameters or token proposals are introduced. Preserve the original identity path byte for
byte and verify all six archived forward/direction controls. Nonidentity rules are new
numerical methods, not execution-equivalent replacements.

For gradient g=Jn^T rhs and p=P g, test quarter-Polyak
delta=.25 ||rhs||^2/(g^T p) p, and one preconditioned normal-equation step
delta=(g^T p)/(||Jn p||^2+lambda||p||^2) p, with
lambda=1e-4 ||g||^2/||rhs||^2. Retain explicit zero/degenerate guards and the existing radius.
First independently qualify CPU FP64/FP32 algebra, SPD construction, and identity equivalence.
Then use a documented resource preflight and largest public context before36public cells:
six contexts x three metrics x two rules, three repeats each. Charge the added matrix
product and all setup. Compare true nonlinear mismatch after a full clipped step, not just
the linear fit. Preserve every result, including failures. No development labels are needed.

A reconstruction grid, selected active rule, execution optimization, and both canonical
setups remain conditional future work. This plan is unimplemented and makes no efficacy claim.

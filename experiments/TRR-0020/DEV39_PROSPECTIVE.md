# Development39 prospective: residual-strength continuation

Development38 completed48reconstruction cells and did not establish a
replacement. Every locked first decision was correct, yet later-token results
were mixed and often worse. Together with the public loss barriers, this
motivates changing the inverse path instead of merely freezing a token.

Define F_lambda by multiplying every attention residual write and every MLP
residual write in the frozen prefix by lambda. F_0 is identity and F_1 is the
unchanged declared FP32 prefix map. Prefix weights never train or change.
For a reconstruction, BOS input stays its known embedding and only the unknown
input vectors move. BOS intermediate states legitimately evolve with lambda;
the entire causal Jacobian must therefore retain past-position effects.

The proposed continuous equation is F_lambda(x)[1:] = observed_H[1:].
At lambda0, x[1:]=H[1:] is an exact root. A predictor direction solves
J_x dx/dlambda = -dF/dlambda over unknown positions, with zero BOS input
tangent. A corrector would solve a new observed residual using the same
full-sequence Jacobian. This is an unproven path: it may fold, become singular,
leave the token manifold or reach an unsuitable continuous root. Final token
readout would scan the entire vocabulary, with no proposals or A2verifier.

Initial implementation is limited to the forward homotopy, analytic full coupled
input/strength JVP and a globally coupled GMRES primitive. This differs from
stage21's own-position diagonal Jacobian. It is not yet a decoder.

Qualify20tiny CPU cases: FP32/64,3/5positions, lambda0/.25/.5/.75/1.
Use independent torch RMSNorm and scaled-dot-product-attention forward plus
dense autograd Jacobians. Check input, strength and joint tangents, causal
zeros, unchanged known-BOS input, exact identity endpoint and exact old
forward endpoint. Verify eight global linear solves against independent dense
solutions and zero RHS preservation. No source labels or model datasets enter.

After CPU algebra passes, qualify the actual public prefix at several lambda
values against independently scaled native modules and finite/dense derivative
references where practical. Estimate GPUcost and memory before execution.
Only then define bounded predictor/corrector budgets and test fixed public
trajectories. No canonical method is registered or selected at this stage.

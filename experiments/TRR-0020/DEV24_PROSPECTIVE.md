# Possible follow-up: match the cosine geometry directly

The baseline ranks tokens by cosine similarity, while the current CGLS reconstruction grid fits raw activation values. Independently of that grid's eventual scores, an untested alternative is to normalize predicted and observed activations and solve the corresponding local least-squares equation.

For unit prediction u=p/||p||, the response is (I-uu^T)J/||p||, and its transpose is J^T(I-uu^T)/||p||. The prototype uses the already-implemented forward/transpose operators with these analytic projections. It retains all vocabulary entries and has no proposer or verifier. Normalization removes a radial output direction, so the CPU checks use positive ridge multipliers1e-4and1e-2and explicitly exercise zero RHS.

A tiny independent dense-autograd check is prepared for both projected operators and the regularized solve. It has not run. No reconstruction grid, GPU execution or efficacy claim is selected. The running development23grid remains frozen and unchanged; any later experiment needs its own plan, resource preflight and complete freeze gate.

The initial positive-norm CPU reference passed. The implementation now explicitly disables the radial derivative below the1e-12normalization floor, matching the clamped denominator, and adds independent zero/tiny-input checks. Rerun and preserve that amended CPU qualification before real-prefix execution.

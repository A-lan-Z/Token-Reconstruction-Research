# Development36 public loss-landscape diagnostic

Development35 identified14/16public identities for all three predefined final
readouts, both update budgets and both observation precisions, matching the
unchanged rule. Norm preservation alone did not solve the "a" and "the" cases.
The norm scaling was real (1.003-3.574), but this is not evidence of an effective
reconstruction improvement.

Before another decoder or optimizer grid, verify native A2's output on every
known public fixture embedding against its full-prefix FP32 observation. This
checks whether the correct endpoint is actually available under the current
causal implementation. BF16 observations are an explicit mismatch condition;
report their endpoint errors without claiming dtype equivalence.

For each of the16public fixtures, two observation dtypes and ordinary/normalized
interpolation, define a diagnostic path p(alpha)=(1-alpha)p0+alpha*one_hot(public_id).
This path uses an explicitly public known identity for diagnosis, never hidden
truth, candidate generation or a reconstruction decision. Evaluate all17fixed
alphas declared in the script, including0and1. Preserve complete input/output
curves, cosine errors and output-normalized squared errors. No adaptive alpha
selection or token reconstruction is performed.

At alpha0, require exact equality to the archived initial output and verify the
directional cosine derivative both from the saved full probability gradient
and an independently qualified input JVP. Record the corresponding squared-error
derivative, true coordinate's initial probability, full-vocabulary logit rank and
full-vocabulary gradient rank. These ranks are descriptive public evidence, not
a vocabulary restriction. At alpha1, verify the input equals the public token
embedding and the output matches the full FP32 public reference within the
existing5e-4relative /5e-5absolute tolerance.

The complete matrix has64curves and1088path forwards plus64initial forwards,
64JVPs,32endpoint checks and one BOS history commit. Run each native one-token
map separately; no padding or batching workaround. Source and public-fixture
hashes, exact commands and resource costs must be retained. A bounded public
curve study does not establish hidden-input accuracy or an impossibility result.

Use the evidence to choose the next scientific question: an initialization
problem, a local loss barrier, objective choice, or an implementation mismatch.
Do not preselect a new reconstruction rule from this diagnostic before seeing
and recording the complete matrix.

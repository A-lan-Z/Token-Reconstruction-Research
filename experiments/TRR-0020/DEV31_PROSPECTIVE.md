# Next public diagnostic: calibrating the mirror step by actual probability change

Development29 and30 completed96reconstruction cells. Neither constant nor observed-error
scaled mirror steps provides adequate quality. The stable warm64/tau.01 rule leaves four
shifted-prose mistakes and one matched-stress mistake. A post-freeze diagnostic shows
all five of those development29 mistakes have maximum token probability below.5 and
observed cosine error above.18. These mistakes are not simply confident wrong final
tokens or small-residual numerical jitter. This observation does not prove their cause.

The current step uses a local KL approximation and a maximum logit-change range of4.
That cap is controlled by the most extreme full-vocabulary gradient values. Test whether
an explicit probability-change budget makes a more useful step without extra A2 calls.
Do not claim the cap is the bottleneck before this diagnostic.

Proposed path: q_eta=softmax(z-eta G), with every vocabulary logit finite. Calibrate eta
using actual forward KL(p||q_eta), rather than only the local weighted gradient variance.
Along this path forward KL is nondecreasing for eta>=0. With centered g=G-E_pG,
K(eta)=logsumexp(log p-eta g) plus the floating-point centering correction; its derivative
is E_pg-E_qg. A finite upper bracket can be derived from the minimum gradient coordinate
and its log probability. This minimum is used only to bound a scalar step length:
it never creates a candidate list, masks vocabulary rows, or selects a token for A2.

Implementation requirements before any GPU use:
- derive and independently test the KL formula, monotonicity and finite bracket on CPU;
- use a fixed safeguarded scalar solve, explicit zero-gradient guards, and a documented
  numerical margin; return a feasible lower bracket and disclose any unused KL budget;
- compare with an independent FP64 root calculation and direct KL from the resulting
  distributions, including concentrated/extreme finite logits and FP32 roundoff;
- preserve every failed reference and do not silently relax tolerance;
- record the added vocabulary sweeps and runtime; this may cost more than the old update.

If CPU checks pass, freeze a public-only24-case comparison: lengths128/40 x Gini warm0/64
x tau.01/.1/1 x original span4/new KL calibration, three repeats each. Use the fixed public
random-token fixture to generate H, then only H for input inference. Derive one full
probability gradient at each warm state, apply each fixed update, and compare actual
nonlinear observed error before/after, exact per-position KL, final confidence, and time.
Require original-control reproduction, largest geometry resource qualification and raw
evidence. Save updated mixture embeddings, their prefix outputs, token readouts, scalar
solve diagnostics and full source/asset bindings. Repeated internal full-logit outputs
must agree; no public token-accuracy result substitutes for reconstruction testing.

This is a public numerical diagnostic plan, currently unimplemented. A reconstruction
grid and any active canonical rule remain conditional. Adding a discreteness penalty to
mirror descent is a separate alternative; Gini with Adam was already studied and must
not be represented as a new mechanism. The full no-shortlist goal is unchanged.

# Development35 probability-mixture geometry

The development34 backtracking rule reduces some public activation errors but
identifies only13/16common public tokens versus14/16for the unchanged rule in
both FP32 and BF16 observation conditions. It also adds prefix evaluations and
fails on a parenthesis that the nonmonotone rule recovers. Do not advance this
rule to a reconstruction grid.

New hypothesis: the ordinary probability-weighted average of embeddings can
have a much smaller norm than actual token embeddings, because vectors cancel.
The optimizer then explores an input geometry the prefix does not normally see.
This is a hypothesis, not an established cause of the failed reconstructions.

Define m=sum(p_i E_i), q=sum(p_i ||E_i||^2), and x=m*sqrt(q/||m||^2), with
both scalar quantities clamped at1e-24. This preserves expected squared norm
away from the zero-mean clamp and equals each token embedding mathematically at
a one-hot distribution. It does not guarantee a unique probability representation.
Every vocabulary probability remains eligible, no token list is proposed, and
no model parameter is trained.

Differentiate through this map exactly to obtain the full probability gradient.
First compare against independent autograd for linear/nonlinear tiny maps,
FP32/64, regular, zero/tiny mean and zero-table cases. Check all synthetic
one-hot vertices and energy preservation separately. Record clamp-edge errors
relative to the potentially large gradient scale.

The planned actual-prefix public matrix reuses all16common-token fixtures and
both observation dtypes from development34. Compare unchanged fixed2/cap1
updates with the new normalized-mixture map, at8and32updates, three repetitions:
128cases. Retain the same four-iteration scalar KL rule and all constants. Require
all64unchanged control outputs to reproduce the old fixed8/fixed32 arrays on both
dtypes. Qualify the new full probability gradient independently through the
HuggingFace prefix at all32initial states.

Record initial raw-mixture norm, expected embedding norm and scale; preserve the
entire final logits, forward outputs, gradient references, KL traces and commits.
Predeclare three final readouts over the entire vocabulary: probability argmax,
Euclidean nearest embedding to x, and prefix-metric nearest embedding to x.
These are final decisions, not lists sent to A2. Charge the two extra vocabulary
products separately. Public commit arrays continue to qualify probability-argmax
history only; any chosen alternative readout requires its own causal decoder
history checks before reconstruction.

Evaluate public identity sanity only after all public cases finish. Public
identities cannot route or stop trajectories. No hidden reconstruction labels
are loaded. A benchmark or replacement claim requires subsequent complete
frozen reconstruction and both canonical setups.

The norm table adds128256FP32values (about.49MiB). Its straightforward computation
temporarily squares the full FP32embedding matrix (about.98GiB); include that
preparation peak in preflight. No attention padding, batching or precision
change is introduced in the prefix. Treat the interpolation as a new numerical
method, not an exact port of the old decoder.

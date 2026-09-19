# Development29 frozen reconstruction plan

Six configurations: Gini warm0/64 x mirror tau.01/.1/1;64mirror updates, full128256 vocabulary.
Each uses the same eight retrospective development records, giving48cells. The largest
warm64/tau1/128-position case qualifies first. New methods remain exploratory; no canonical
method is active until a promising fixed decision rule is selected.

Reuse one DiscreteSoftVocabulary instance with separate warm/mirror graphs and a shared
owned capture stream. The warm stage is the original Gini003 rule and original initialization.
The mirror stage uses the existing power1 probability-gradient backward and the qualified
mirror_step.py update. No Adam moments or logit decay operate during the mirror stage.
Reset only mirror diagnostics after the warm stage; preserve the complete warm logits.
The full-vocabulary initial mixture must match dev26 exactly, and every warm trajectory
must match dev7. No new fitted predictor or discrete candidate verifier is present.

Before each configuration's real records, run three exactly repeated synthetic128/40
whole decodes and one eager mirror decode per length. Require output tensors, warm traces
and mirror loss/confidence/Gini traces to be exact. Independently compute the probability
cotangent as embedding cotangent times E^T, then center it, and require exact agreement
with the power1 backward. This validates the declared BF16 surrogate gradient, not finite
differences through quantization. Synthetic IDs generate observations but never enter decode.

Save final tokens and fixed snapshots0/8/16/32/64, whole-objective best tokens and the
diagnostic per-position best tokens. These selection rules use only observed continuous
activation error. No reranker repairs a mixed-context per-position result. Save initial
mixture, update diagnostics, final confidence/error, all timing and source/asset bindings.
Checkpoint times are observed times after the corresponding training call; snapshots
before the final evaluation include one extra update already executed and must not be
presented as minimal standalone checkpoint runtimes.

Work per input is warm+66 whole-prefix forwards and warm+64 backwards, plus one saved
initial-mixture product. Report prefix/table construction, both graph captures, warm stage,
mirror stage, synchronization, transfers and disk I/O. First-input capture is included in
total inference and reported separately. No batching or precision substitution is allowed.

Freeze all48cells and source hashes, then require complete phase qualifications, exact
warm/mean anchors, valid finite outputs, full-vocabulary/no-verifier work counts, and
negative missing/source/output gates before current retrospective truth opening. Archive
all raw outputs and repeated/eager qualifications. No overall replacement claim can be
made without both canonical setups and the complete active matrix.

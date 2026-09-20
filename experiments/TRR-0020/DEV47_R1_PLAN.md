# Development47 revision1: bounded one-history iteration mixing
Unqualified exploratory algorithm change. No shortlist, token verifier, model
fitting or vocabulary restriction. Aim to reduce prefix forward/backward steps,
not merely shave execution overhead from the same inaccurate trajectory.

Anderson mixing combines recent fixed-point outputs using coefficients found
from residuals. Reference: Walker and Ni, SIAM J.Numer.Anal.49(2011),1715-1735,
https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf .
Our bounded/logit-coordinate/probability-residual adaptations are new empirical
rules; no convergence guarantee from that paper is claimed.

Let F be the qualified full-vocabulary KL update. Keep one previous proposal
and residual. Compute gamma=dot(r,r-r_old)/(norm(r-r_old)^2+
1e-4*(norm(r)^2+norm(r_old)^2)); clip globally to[-c,c]. Mix logit proposal
F(x)-gamma*(F(x)-F_old), then remove its row maximum. First step or zero
denominator uses the exact unmodified proposal. Reset history each record.
Residual is either F(x)-x (logit) or softmax(F(x))-softmax(x) (probability).
The latter retains a probability metric but mixes logits to preserve valid
softmax distributions. Every vocabulary coordinate remains eligible.

Four frozen public configurations: unchanged reused-probability control,
logit clip1, probability clip1, probability clip2. Fixed64 updates, same
seed200051 and native lengths128/40, three repeats plus one eager check per
mode/length. Largest probability clip2/128 first. Independent gradient checks;
unchanged control reproduces all archived outputs/traces, all other modes
reproduce warm initialization. Full arrays and traces retained. History mixing
does not preserve the original KL bound; base-update diagnostics are explicitly
labelled as base diagnostics. No adaptive correctness routing.

CPU qualification compares gamma and mixed outputs with an independent
constrained two-column KKT solve, including degenerate and no-history states.
GPU public numerical diagnostics freeze all8cells before calculating token
identity curves from already known public fixtures. Timings are actual64-step
cost only; intermediate checkpoints are not independently stopped runtimes.
This is not a fresh or canonical benchmark. A promising rule needs a separately
frozen full development matrix and, if selected, both canonical benchmarks.

Preflight: full decoder base peak4.58GiB; two persistent127x128256 FP32history
arrays add124.3MiB. Largest expected peak<=7GiB including temporary arrays and
graph pools. Require10GiB initial GPUfree, no other compute processes; guard
retains2GiB GPUfree,8GiB hostfree,RSS<=10GiB,temp<80C,timeout900s. Inner guard
retains3GiBfree/reserved<=8GiB. Estimated public pipeline<=240s.

Revision1 integration correction: the original subclass used self.metric and
self.name, which the parent already owns. The parent replaced metric with its
PrefixWeightMetric object; the object compared unequal to the string logit,
and receipt JSON serialization then failed. The first requested probability
clip2 worker produced raw arrays, but no complete valid qualification receipt.
The entire initial attempt is preserved and excluded. No other mode ran.
Use history_metric/history_name/history_clip attributes. The mathematical
mixing function is unchanged. Rerun qualification under distinct paths and
source bindings; do not overwrite the failed output or edit frozen old source.

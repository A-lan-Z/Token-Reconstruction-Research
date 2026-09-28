# Development49 revision1: full-vocabulary steps from moment bounds

Exploratory public numerical qualification; no new active reconstruction method.
The current four-iteration full-vocabulary KL root solve remains expensive.
Earlier tests compare4/8/16 iterations and span clipping, but no moment bound.
Test two bounds requiring only per-position gradient moments, with all128256
vocabulary coordinates retained. No candidate generator or verifier is added.

For normalized gradient d and current distribution p, let X=E_p[d]-d,
v=Var_p(d), and b=E_p[d]-min(d). The KL of the exponential-path update is
log E_p exp(tX). Test (1) log(1+(v/b^2)*(exp(bt)-1-bt)); and (2) the sharper
two-point upper bound log((v*exp(bt)+b^2*exp(-tv/b))/(v+b^2)).
The second follows from Corollary1(i), equation5, Seneta and Weber,
Attainable bounds for expectations (1982), DOI10.1017/S144678870001884X:
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F2D9D5BB90840A4C10A044C66D48877F/S144678870001884Xa.pdf/attainable-bounds-for-expectations.pdf
These are mathematical upper bounds under exact arithmetic, not a claim that
the finite-precision implementation is globally certified. Validate actual KL
independently. No convergence or token-recovery theorem is claimed.

Budget remains clamp(2*observed_cosine_error,0,1), margin1e-4.
Normalize the cached FP32 probability mass for moments; retain original native
reduction geometry and exact elementwise kernels. Compute scalar roots in FP64
with48fixed bisections, then shrink rate by1e-6 and cast to input precision.
Use a zero step when variance, b, span or budget is <= input dtype tiny.
A stable exponential-remainder evaluation handles small arguments.
The new proposal uses the original logits minus rate*centered gradient,
followed by row-max subtraction; it is a new numerical decision rule.

CPU matrix: FP32/64 x vocab17/37/1031 x7fixed fixture types x2bounds=84cases.
Seed49049+vocab; negative/zero/tiny/ordinary budgets in each fixture.
Independent80digit mpmath scalar roots, direct FP64 distribution KL,
finite values, unchanged inputs, and degenerate no-op checks.
GPU matrix:21FP32 tiny fixtures plus8frozen original public states spanning
0/16/64/128updates at native128/40lengths, each new rule repeated3times.
Compare scalar kernels to the torch FP64 implementation and independent KL;
require relative scalar-root error<=3e-6, probability TV<=2e-5, direct KL
<= requested budget+2e-5, exact repeat/eager-graph checks and unchanged inputs.
Verify original control outputs against every immutable public fixture.
Time three methods on each public state,3rotating groups x20replays.
Kernel timings are not reconstruction speed or quality. Save full update
arrays and all scalar diagnostics. Freeze sources and plan before qualification.

Largest127x128256 geometry first, no padding or microbatching.
Live RTX5080 has11109MiBfree, no competing compute,39C; host free>29GiB.
Estimate<=6GiB reserved for qualification (no prefix model loaded), vs
prior20case exact-update qualification2.635GiB. Require>=10GiBinitialfree,
guard>=2GiBfree/RSS<=10GiB/host>=8GiB/temp<80C; inner>=3GiBfree/reserved<=8GiB.
Tensor arrays are65.2MB each; original controls, new intermediates, FP64 direct
KL and three captured implementations are included in the conservative bound.
Runtime estimate<=180s including compilation and1440timed calls; timeout600s.
Only advance to a separately frozen full decoder after numerical qualification.

The original GPU launch failed before calculation through a generic cpu_check import collision. Revision1 gives the helper a unique module name and appends the shared pointwise path. Arithmetic is unchanged; rerun all CPU and GPU qualification under separate create-only paths. The original attempt remains excluded and preserved.

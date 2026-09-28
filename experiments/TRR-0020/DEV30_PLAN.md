# Error-scaled full-vocabulary mirror descent

Development29 completed its48-cell matrix and all controls. Cold tau.01 at64steps recovers
248/254 matched prose and246/254 shifted prose, but only68/78 and67/78 stress tokens.
Warm64/tau.01 reaches254/254,77/78,250/254,78/78 at about1.033s per128-position input.
Larger tau frequently loses prose accuracy after earlier gains. Constant normalized step
sizes need not vanish near a small observed residual. This motivates an explicit change,
not a claim that scaling will solve the remaining errors.

Keep the qualified mirror rule and all six warm0/64 x tau.01/.1/1 configurations, with64
mirror updates and checkpoints0/8/16/32/64. Multiply the entire mirror rate by
sqrt(clamp(current per-position observed cosine error/.05,0,1)), with no positive floor.
Thus both the local KL approximation and maximum logit change shrink near zero error.
The .05 normalization is inherited from the established warm optimizer; it is not fitted
on new labels. Negative floating-point cosine roundoff produces zero scale. Prefix weights
remain frozen and all128256 vocabulary rows remain eligible. There is no A1 or shortlist.

Qualify algebra, exact unit-scale equivalence, zero-error/negative-roundoff handling,
actual-prefix probability gradient, three repeated public128/40 decodes, and eager/replay
equality before each configuration's reconstruction. Require the same exact dev7 warm
and dev26 initial-mixture anchors for all48cells. Inherit development29 timing/work-count
and checkpoint conventions. Freeze all48outputs and source/assets before retrospective
scoring; preserve all failed runs and test the three negative truth gates. The prior
development29 matrix remains immutable and is not represented as a fresh paired baseline.
No active canonical method is selected before this exploratory result.

# Prototype amendment before second independent synthetic cohort

First fixed-cohort result (seeds100-102, 48 random sequences): sequential369/384,
diagonal proposal control372/384, full joint379/384. Joint is better at token
recovery but uses much more forward/reverse work. These results are preserved.

The next phase is an explicit new implementation and decision-policy change:
- Production joint gradient uses one reverse pass through sum of block losses,
  rather than the per-position diagnostic gradient decomposition.
- Each solver checks whether the currently evaluated discrete hypotheses already
  match before taking another backward step. Apply this to ALL relevant methods.
- The block solver commits the complete block only when its actual forward loss
  meets a fixed matched-FP64 criterion (sum normalized loss <=1e-20). Otherwise it
  commits only the first token and reuses the provisional suffix. No guarantee
  is claimed under model mismatch or arbitrary finite precision.
- Keep window3, beam2, proposals2, rounds4 unchanged.
- The original cohort is now development/replay evidence for this amendment.
- New cohorts use model seeds200,201,202,16 sequences/model,8 tokens/sequence.
- Include both the optimized sequential control and optimized block-verification
  with diagonal-only proposal gradients. The latter has a more expensive gradient
  decomposition solely to isolate the information intervention; do not infer
  production acceleration from its CPU timing.
- Hold target/noise intervention cases separate from the primary matched case.

No parameter search or oracle-based fallback. New outputs will be separately named.

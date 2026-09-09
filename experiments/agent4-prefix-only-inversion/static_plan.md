# Bounded static-surrogate fallback

After matched-model numerical feasibility (actual gradients, exact raw-input
paths,18/21 development tokens), run the unchanged frozen variant on two static
mismatch records where practical: the first ordinary and first stress final
source by selection order. Compare with the same public-prefix A1+A2 comparator.
No source/correctness-dependent selection. No further optimization variants.

Target: existing public Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct checkpoint
7fa9d06a59246629244cdd3b6b92e4fc756baa0f, evaluator-only. Model card identifies
Llama-3.2-1B-Instruct as base and SFT on GrandMaster-PRO-MAX. Capture through four
blocks after casting its published fp16 weights to BF16. Verify vocabulary
identity and actual clean restore. No new target training or trajectory sweep.
Primary solver still receives only the original public base approximation.

This is not a recovered-prefix test. No verified prefix-recovery implementation
or state is available in the pinned branch. Messages requested such assets from
Agent2; do not substitute its private target adapter as recovery. Stage3 remains
not run unless such a legitimate path appears and Stage2 becomes promising.

Same16position preflight geometry; one-off target backup/capture separately
recorded. Reuse frozen128checks/steps and best-tried policy, which is allowed
under mismatch but does not certify correctness through low residuals.

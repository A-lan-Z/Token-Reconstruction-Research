# Joint-activation proposal prototype: fixed synthetic plan

This is a CPU mechanism test, not a Llama benchmark or a run on Agent 4's data.
No decoder weights are fitted. The local Agent 4 rescue source is not available.

## Proposed algorithm
Discrete rolling-window search. Keep a small beam of actual token blocks; use
observed block residuals to differentiate through the frozen causal forward map.
Use vocabulary-wide first-order/proximal scores to propose NEW token substitutions.
Batch actual discrete verification, retain the best blocks, commit only the first
position after a finite search budget, then roll forward. Later guesses are
provisional and are never supplied as true labels.

The primary mechanism ablation uses the same block engine with gradients detached
through the strictly-earlier source positions for each observation. Thus later
observations still verify blocks, but cannot generate earlier-token proposals.
A current-position-only sequential solver supplies the stronger cost control.

## Tests before any model-specific performance claim
1. Exact constructed causal map demonstrates a root missing from local proposals
   entering through later-observation gradients, with no future-token labels.
2. Continuous-nuisance cancellation: a square causal map with invertible future
   diagonal can locally absorb earlier-token changes if future inputs are free.
3. Exhaustive small vocabulary reference and cache/autodiff functional checks.
4. One fixed small random causal transformer family (not pretrained): 4 blocks,
   width 32, 4 heads, vocabulary 128, 8 unknown tokens with known BOS. One
   development seed, followed by independent seeds 100,101,102 with 16 sequences
   each. Report all seeds and failures, not just successful examples.
5. Compare identical proposal-engine controls and a current-position cost control.
   Count actual forward/backward calls, transformed positions, candidate blocks,
   vocabulary scans. CPU wall time is illustrative only, not GPU prediction.

## Initial bounded settings
Window 3, beam 2, 2 proposed replacements per position, at most 4 rounds/window.
All positions initialized from fitting-free raw activation/embedding similarity;
recycle provisional suffix on sliding the window. Use token-specific input
embeddings, not normalized vectors when executing the forward model. No special
truth-assisted rescue. If early implementation/prototype choices change, record
those changes before independent-seed runs; do not hide failed development runs.

## Decision
No large Llama experiment is authorized by toy gains alone. A later handoff must
state exact necessary cache semantics, a finite work cap, and a first-stage gate:
new correct proposals under a fixed forward-equivalent budget. If the prototype
fails to provide a cost case, report that and limit any proposed native experiment
to a bounded mechanism check rather than a rescue campaign.

# Development34 full-vocabulary descent qualification

The complete post-freeze causal audit is dev34_causal_error_audit.json.
For factor2 / eight updates, all eight first mistakes occur with substantial
remaining mixture error (.144-.516) and confidence .121-.667. Every first
mistake's path contains two or four increases in observed activation loss.
The prefix before each first mistake is correct by definition. This rules out
past-token error propagation as the sole explanation for those first errors,
without proving that backtracking will recover their tokens.

The new rule is Armijo backtracking over full-vocabulary probability states.
Use the existing FP32 current-token map/VJP and four-iteration forward-KL path.
Start with budget min(2*observed cosine error,1). For each of a fixed number
of updates, compute one gradient at the current accepted state. Try at most
four full-vocabulary exponential steps, quartering the budget after a rejected
trial. Accept only when the observed trial loss is at most the old loss plus
1e-4 times the full probability-gradient dot probability change, and that
linear change is nonpositive. Reuse the accepted forward cache for the next
gradient. After acceptance, grow the successful budget by at most2, capped by
twice the new observed error and by1. If all four trials fail, retain the old
state and carry the quartered budget into the next update.

All128256 vocabulary logits remain finite and eligible. Trials are continuous
probability mixtures, not a shortlist of token IDs; no separate token verifier
is introduced. Only the final argmax is emitted. The actual-token forward
after emission exists only to commit history. No weights are trained.

First qualify the update on independent tiny nonlinear autograd references,
check state/cache consistency, exercise rejection, and verify monotonic
accepted loss plus direct FP64 KL feasibility. Then test actual public-prefix
fixtures, with saved source hashes and no reconstruction labels. The frozen
public matrix is twelve contexts x two rules x8/32updates =48cases, each with
three exact repetitions. Six contexts are the unchanged development33
fixtures at lengths128/40 and last/middle/first unknown positions. Six new
contexts use public synthetic IDs from seed34021+length, known BOS, uniform
sampling over the complete vocabulary and the same three positions.

The control is the unchanged factor2/cap1 fixed-step development33 trajectory.
Its old eight-update outputs must match every saved tensor on all six old
contexts. For all twelve contexts, independently compare the full-vocabulary
probability gradient with autograd through the HuggingFace prefix at fixed
synthetic history. Preserve gradients, forward outputs, final full logits,
mixtures, traces, direct FP64 KL for every trial, and emitted commit outputs.
No hidden-input reconstruction or accuracy claim is made from these cases.

Run the largest native new128-position context first with32descent updates,
then the entire matrix once numerical/resource gates pass. Estimate transient
storage for at most128trial states plus duplicate old logits. Native attention
lengths and FP32 arithmetic are unchanged. Record all preparation, inference,
commit, validation, I/O and resource costs. Require the external idle GPU
guard and internal8GiB reserved/3GiB GPU free/8GiB host available margins.
Any failure is preserved and terminal before repair or rerun.

If this reduces difficult public activation errors at useful evaluation cost,
qualify exact replay and define a full causal decoder matrix. Otherwise retain
the negative result and reassess. This exploratory public diagnostic does not
register an active canonical method. No threshold is selected from token
correctness, and all future evaluation outputs still freeze before labels.

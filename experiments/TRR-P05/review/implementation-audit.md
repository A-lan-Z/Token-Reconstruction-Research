# TRR-P05 implementation audit

Audit scope: model-free inspection of the current P05 implementation and
focused synthetic tests. No model, truth file, prediction row, or diagnostic
run was opened. The runnable script is the implementation-owned
`scripts/trr0004_p05_diagnostic.py`; this review does not add a competing
runner.

## Passed contracts

- The declared sample is 384 cached teacher rows plus 384 deterministic
  same-pool controls. Controls carry no teacher scores or fabricated rank
  labels.
- Gradient batches come from the four original P04 schedule steps per seed,
  with the recorded 6 replay + 2 correction rows and sparse teacher mask.
  Gradients are measured at selected H, selected D, and final D, with no
  optimizer call or parameter update.
- The implementation calls the frozen P04 CE, hard, and ranking objectives;
  the gradient receipt records active rows, retained pairs, pair-weight sum,
  clipping, and the negative-gold-margin diagnostic. The positive cosine
  interpretation is documented with the required sign.
- The affine initial diagnostic is an evaluated PR7 public affine function;
  no unrecorded GRU initial state or trajectory is recreated.

## Required fixes before GPU execution

1. **Prediction tie rule.** `_top2_metrics` (currently lines 611–618) uses
   `torch.topk(...)[..., 0]` as the predicted token. `topk` does not provide
   the declared lowest-token-ID tie rule, so exact ties can change accuracy
   and correctness rows. Use `logits.argmax(dim=1)` for the decision and
   retain the top-two values only for the gold-margin calculation. The
   gradient negative-margin helper should use the same deterministic policy
   where a selected subgradient is required.

2. **Exact forward rank aggregate.** `_row_rank_metric` computes the global
   weighted numerator and denominator, but `forward_state` discards that
   aggregate and writes only per-row rank means. Since per-row pair weights
   are not retained, `teacher_rank_loss_row_mean` cannot reproduce P04's
   global `sum(weight * softplus) / sum(weight)` loss. Persist a sample/group
   aggregate with the exact numerator, denominator, retained pairs, and
   omitted ties (or explicitly fail if it cannot be produced); do not publish
   the row mean as the original P04 loss.

3. **Candidate provenance binding.** `_validate_candidate_binding` receives
   `candidate_metadata` nowhere and checks only tensor shape plus the 384
   teacher rows. Before any H/D gradient, require candidate metadata
   `pool_record_order_sha256`, `pool_observation_sha256`,
   `pool_records_sha256`, and candidate K to match the combined public pool,
   and bind the candidate artifact to the recorded embedding/proposer inputs
   when those hashes are available. Shape and teacher-row agreement alone
   would permit a reordered or otherwise mismatched replay portion.

4. **Fixed state count.** `collect_state_specs` currently tolerates missing
   final files after checking only the six selected states. The approved P05
   scope is exactly one affine reference plus all six selected and all six
   final S/H/D states; fail closed if any of the 12 stored states is absent.

These are pre-run correctness/provenance issues, not requests for a larger
sample, a new objective, or a new experiment. The current public schedule
metadata has teacher rows in every predeclared batch, so no conditional batch
replacement is needed for this frozen input.

## Test status

The existing focused synthetic tests cover non-contiguous candidate gathers,
pool-row ordering, and no-update H gradients. They do not cover the required
lowest-ID full-vocabulary tie decision, exact forward rank aggregation, or
candidate metadata mismatch; implementation should add the smallest tests for
the fixes above before the resource-qualified run.


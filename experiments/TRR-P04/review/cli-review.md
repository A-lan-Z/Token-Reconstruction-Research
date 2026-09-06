# TRR-P04 CLI and preflight review

**Review status:** public capacity qualification passed on 2026-09-06 under
source commit `628c843ee127c0b5014803f8226d8f795c0c4579`. The r5 receipt includes
full-pool/probe gate results, GPU free/peak memory, host RSS, UTC timestamps,
and input hashes. No evaluator truth, target-update weights, source rows, or
matrix-training jobs were opened during this review.

## Checks that now pass

- `trr0004_p04_predict.py` now emits the scorer schema
  `token-reconstruction.trr-p04-predictions.v1`, rejects source/truth fields,
  validates finite activation and right-padded mask geometry, and produces
  full-vocabulary predictions with a separate tie diagnostic.
- `train_arm` initializes the validation metric from step zero and retains the
  initial state on regression. The focused P04 suite covers this regression and
  the prediction CLI schema round trip.
- The current `public_selection-r2.json` source descriptors contain the pinned
  Pile, Finance, and Alpaca hashes; the earlier null Pile hash is no longer an
  active setup discrepancy.
- A tiny synthetic CPU check passed: changing future activation rows leaves all
  earlier predictions bit-identical, confirming the one-layer GRU path is causal
  under the current prediction helper.
- The canonical candidate-preparation CLI and shared-artifact consumer now exist;
  the trainer rejects missing or mismatched candidate metadata before fitting.
  The P04 training proposer declares descending-score/ascending-token-ID ties;
  native anchor ties remain separate. The existing r1 candidate artifact was
  generated before that metadata field was added and must be regenerated or
  explicitly bound before teacher/H/D consumption.

## Execution-contract findings

1. **Qualifier gate and resource evidence pass.**
   The corrected `scripts/trr0004_p04_qualify.py` applies ≥256 initial errors
   and <0.99 accuracy to the full 256-record correction pool, while the exact
   eight-row capacity probe requires actual initial errors and post-update
   improvement. The r5 receipt reports 4,185/45,596 full-pool errors, 1,000/1,503
   probe errors, and token accuracy 0.334664 → 0.806387. Its GPU/host guard also
   passes: 15.679 GB free preflight, 12.840 GB free post-run, 2.512 GB peak
   allocated, 2.749 GB peak reserved, and 3.891 GB peak host RSS.

2. **One shared candidate artifact is wired end to end.**
   `scripts/trr0004_p04_prepare_candidates.py` writes `candidate_ids`,
   `proposal_ids`, and confidence with pool/embedding/affine hashes; teacher
   qualification and H/D training consume those rows rather than regenerating
   candidates. The P04 training proposer may use its declared ascending-token-ID
   tie rule; it need not reproduce native PR7 `torch.topk` tie ordering. The
   artifact and receipt must state that rule, while native A1+A2 anchor ties
   remain separate. Teacher/training loading should still bind the artifact's
   affine/table/pool hashes, not only shape, IDs, and embedding-file equality.

3. **Proposal budgets need one frozen naming convention.**
   Candidate preparation stores `proposal_k=512` and `a1_ranked_k=512`, while the
   current teacher metadata still writes `proposal_k=32` plus `a1_ranked_k=512`;
   make the K=512 proposal and K=32 retained candidate explicit and consistent
   across setup, teacher evidence, and training receipts.

## Required strict invocation checks

The scorer/freezer do require all four method arms, both paired seeds, both
target conditions, and both native anchor groups, and they validate every
prediction vector against the panel length. The CLIs still accept arbitrary
same-row-count record manifests, method/seed labels, and optional subsets
(`--arms`, `--seed`). The prediction loader also does not compare its `--seed`
argument with the seed stored in the safetensors state metadata. Before any
fresh prediction or truth gate, the root protocol must bind the exact frozen
plan: ordered record IDs and lengths, observation/mask/position geometry,
public-base affine/table hashes, state method/seed metadata, all four arms and
both seeds, and both target roots plus the native A1+A2 anchor. The prediction loader checks local geometry but cannot establish those
cross-file identities by itself. Likewise, reject teacher evidence unless its
qualification receipt is `PASS`, has exactly 256 difficult plus 128 audit rows,
and binds the candidate/source hashes. A failed teacher gate may produce the
predeclared D-not-run outcome only; it must not be treated as valid D training.

The schedule currently enforces the declared 6 replay + 2 correction record
mixture and retains required teacher positions. Its receipt still needs actual
replay/correction token exposures and per-record repeat counts, as required by
the plan; token proportions need not equal 75/25.

## Test coverage gap

The focused suite passes (`17 passed` after the qualifier/source edits), including
step-zero, prediction-schema, and synthetic shared-row binding checks. Runtime
qualification is now evidenced by the r5 receipt; the main remaining artifact
issue is that candidate-preparation r1 predates the required tie-policy metadata,
so it cannot be consumed by the current strict loaders without regeneration or
an explicit bound compatibility record.


# TRR-P04 CLI and preflight review

**Review status:** implementation fixes applied; qualification remains
resource-gated. This is a static review of the teacher, qualifier, prediction,
and training entry points on 2026-09-06. No evaluator truth, target-update
weights, source rows, or heavy jobs were opened during this review.

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

## Applied execution-contract fixes

1. **Qualifier gate applies to the actual probe.**
   `scripts/trr0004_p04_qualify.py` now chooses exactly eight deterministic
   correction records with initial affine errors, measures that probe's active
   positions, and fails closed unless it has at least 256 wrong positions and
   initial accuracy below 0.99. The all-correction error count remains a
   diagnostic. This is a selection-feasibility gate, not a method-family
   conclusion.

2. **One canonical candidate artifact is wired end to end.**
   `scripts/trr0004_p04_prepare_candidates.py` evaluates the frozen PR7 affine
   proposer once and writes `candidate_ids`, `proposal_ids`, and confidence with
   pool/embedding/affine hashes. Teacher qualification consumes those exact
   rows by record ID. H/D training requires the artifact, validates its schema,
   proposer identity, pool order/observations, embedding hash, and candidate
   uniqueness, and no longer regenerates candidates.

3. **Proposal budgets are explicit.**
   Candidate preparation records `a1_ranked_k=512` through its `proposal_k=512`
   tensor and the K=32 candidate prefix; teacher evidence records
   `a1_ranked_k=512` and `candidate_k=32`. The two budgets are kept separate in
   receipts and diagnostics.

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

The focused suite currently passes (`17 passed` after the latest candidate
wiring edits), including the step-zero and prediction-schema checks and a
synthetic shared-row binding check. It has no bounded test for the qualifier's
probe gate, the candidate-preparation metadata/proposer contract, teacher
qualification gate binding, or a PR7/new-affine source-equivalence check. Add only lightweight synthetic/fixture checks for these contracts before
the public qualification command; no full-vocabulary or teacher-model fixture
is needed.


# TRR-P04 CLI and preflight review

**Review status:** implementation fixes are partly applied; public qualification
remains pending a gate correction. This is a static review of the teacher,
qualifier, prediction, and training entry points on 2026-09-06. No evaluator
truth, target-update weights, source rows, or heavy jobs were opened during
this review.

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
  The shared-artifact wiring is present; its P04 lowest-token-ID tie policy
  must be declared in the artifact/receipt, while the native anchor retains its
  separate native tie behavior.

## Execution-contract findings

1. **Qualifier gate correction is still pending.**
   The visible `scripts/trr0004_p04_qualify.py` chooses eight deterministic
   correction records, but still applies the ≥256-error and <0.99 criteria to
   that eight-row probe. The ≥256 initial-error requirement belongs to the full
   256-record correction pool for teacher-selection feasibility. The capacity
   probe needs actual initial errors and measurable post-update improvement; it
   does not need 256 probe errors. The full-pool gate and probe improvement
   check must be recorded separately and fail closed.

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

The focused suite currently passes (`17 passed` after the latest candidate
wiring edits), including the step-zero and prediction-schema checks and a
synthetic shared-row binding check. It still has no bounded test for the
qualifier full-pool/probe split, candidate-preparation metadata/tie contract,
or teacher qualification source binding. Add only lightweight synthetic/fixture
checks for these contracts before the public qualification command; no
full-vocabulary or teacher-model fixture is needed.


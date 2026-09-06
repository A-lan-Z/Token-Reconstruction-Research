# TRR-0008 evaluation interface

This task owns a small adapter around the reviewed TRR-0007 decoder path. It
keeps the four scientific methods fixed and leaves source selection, public
capture, truth opening, and the final score decision to the root-owned freeze
process.

## Frozen method set

`METHOD_ORDER` in `scripts/trr0008_eval_contract.py` is exactly:

1. `trr6__enriched_trained_diagonal_attention128` — retained reference, loaded
   by `token_reconstruction.trr0005_joint_decoder.load_decoder_state`;
2. `current_enriched__residual_mlp512` — current-bank residual MLP-512;
3. `improved_public_bank__residual_mlp512` — primary candidate;
4. `improved_public_bank__trained_diagonal` — weaker same-bank diagnostic.

`current_enriched__trained_diagonal` is recorded only as
`TIMING_CONTROL_METHOD_ID` for the identical-weight timing control. It is not
in the scientific method order, prediction score matrix, or promotion rule.
There is no A1/A2 default, candidate search, token history, or teacher input.

The selected state paths and SHA-256 values are read from the reviewed
`experiments/TRR-0007/method_freeze.json`; the binder checks the actual files
again. The shared normalized public embedding table is bound by its reviewed
SHA-256. The model source loaders are unchanged:
`token_reconstruction.trr0005_joint_decoder` for the retained reference and
`token_reconstruction.trr0007_positionwise` for the three positionwise rows.

## Loader and prediction API

The evaluator calls:

```python
from scripts.trr0008_eval_runner import predict_current_h
ids = predict_current_h(
    model,                 # frozen torch.nn.Module from the registration loader
    normalized_public_E,   # [128256, 2048], float32
    activation,            # one row H, [128, 2048], usually BF16
    valid_mask,            # [128], bool; all positions valid and BOS at 0
    device=device,
)
```

The return value is a CPU `torch.int64` vector of shape `[128]`. Position zero
is fixed to BOS ID `128000`; positions 1--127 are the full-vocabulary argmax
from the current hidden state only. The function has no access to source IDs,
token labels, earlier hidden states, prefixes, candidate lists, or truth.

The registered models expose `projected_hidden` and `logits_from_rows`; the
runner uses those row-wise methods to avoid materializing a full clip by
vocabulary matrix. It synchronizes immediately around the timed call and
checks finite logits and exact warmup/measured ID equality.

## Observation and panel contract

The four paired cells remain ordered as:

`pile__public_base`, `pile__public_lora_2601`, `finance__public_base`,
`finance__public_lora_2601`.

Each observation artifact contains `activations`, `attention_mask`, and
`position_ids`, with `[records, 128, 2048]` activation geometry, BF16
activations, full-valid masks, and positions `0..127`. The registration binds
`records_by_domain` from the observation manifest. Existing 128-row TRR-0007
observations are accepted for CPU/timing qualification, but fresh TRR-0008
Finance and Pile counts are not hardcoded in the adapter and must be frozen by
the prospective plan. Paired target cells within a domain share their count;
different domain counts are supported.

The public-only binder and runner write create-only registration, prediction,
timing, and run-manifest artifacts. They mark `truth_opened=false` and
`candidate_arrays_persisted=false`. The scorer accepts already materialized
prediction and truth tensors only after the root-owned freeze gate; it does
not perform source selection or infer a truth path.

## Timing qualification

The canonical timing execution path is `scripts/trr0008_timing.py`. It owns the
balanced order, synchronized warmup/measured boundary, archived prediction
equivalence check, resource guard, and precision40 qualification. The final
truth-free receipt is
`experiments/TRR-0008/timing/precision40_result.json` (40 blocks, all four
cells PASS, threshold 1.25). `scripts/trr0008_eval_timing.py` retains only
lightweight schedule helpers for tests; its execution entry point is retired
and cannot create a competing timing receipt.

The registration binds the canonical timing plan/receipt when supplied. The
public gate revalidates all 16 prediction artifacts, per-cell timing records,
registration/observation/run hashes, and the canonical timing receipt before
truth.

No timing or fresh prediction run is authorized by this interface document.

## Prospective score contract

`scripts/trr0008_score.py` reports token accuracy, exact 127-token clip
recovery, paired record gains/losses, and paired candidate/reference
contrasts separately for all four cells. The primary candidate is compared to
the retained reference on Finance public-base exact recovery. Finance shifted
and Pile target cells are paired safeguards, while current residual and
improved trained-diagonal contrasts are descriptive controls. These cells are
not pooled as independent sources.

`proposed_decision_contract()` is only a blocked pointer. The CLI requires the
owner-frozen nested contract before any truth access and consumes its practical
margins, exact CP component tails, one-sided record-bootstrap tails, seed,
draws, four-cell safeguard set, and canonical timing binding. It uses the
correct upper exact bound `beta.ppf(1-alpha_component, k+1, n-k)` and lower
bound `beta.ppf(alpha_component, k, n-k+1)`, including the zero/all-success
endpoints. Token intervals resample per-record token-accuracy means. The
scorer verifies the registered prediction root, frozen run hash, truth sidecar
bytes/hash, and sidecar metadata before loading label tensors, then serializes
the result before its create-only write.

The score path does not recreate TRR-0007's factorial primary family and does
not include an A2 anchor by default. If the frozen quality and cost criteria
are not met, the decision status retains the reference or reports an
unresolved practical benefit.

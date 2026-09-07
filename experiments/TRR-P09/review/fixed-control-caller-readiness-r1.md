# P09 fixed-control caller readiness r1

Status: CPU-only implementation/readiness checkpoint. No public bank, model,
activation, source row, final truth, GPU, or optimizer fit was opened by this
checkpoint.

The shared runner already owns the training loop, lazy `BatchSource`, full-
vocabulary scoring, checkpoint step-zero eligibility, earliest strict maximum,
domain-balanced metric validation, and process timing. The new caller module
`scripts/trr_p09/fixed_control_caller.py` supplies the missing fixed-arm
plumbing without duplicating Agent-1's directional fit or caller helpers.

## Bound interfaces

- `join_public_validation_labels` accepts metadata-only rows
  `{global_row, record_id, domain}`. It rejects duplicate identities, unknown
  or empty domains, and missing required domains, then emits a canonical
  label-join SHA. `make_domain_validation_callback` uses that immutable join to
  request one public batch view per domain and computes the existing equal-
  domain `domain_balanced_token_accuracy` metric through the common runner.
  Token labels remain in the public validation batch source; no truth loader is
  introduced.
- `inherited_schedule_steps` reproduces the registered CPU
  `torch.Generator`/`randperm`/`randint` ordering, including post-BOS filtering
  and replacement flags, while preserving global row identity. The helper is
  lazy; production may consume a bound serialized schedule instead of holding
  all steps in Python. `materialize_schedule_plan` is for bounded receipts and
  fixtures only.
- `signed_p09_checkpoint_grid` binds the signed
  `N=max(12000,1000*ceil(5*actual_valid_positions/(512*1000)))` formula and
  milestones `[0,1000,2000,4000,8000,12000,N]`; it rejects a changed position
  budget.
- `make_fixed_checkpoint_callback` writes decoder-only safetensors once per
  checkpoint, verifies the state digest before/after serialization, and records
  the supplied corrected bank SHA, base-state SHA, fit-manifest SHA, embedding
  SHA, and schedule SHA. It does not write optimizer state or mutate the live
  model. `build_fixed_control_receipt` preserves the runner learning curve and
  exposure summary and adds explicit update/draw/validation/timing costs.

The corrected bank manifest SHA remains a required caller argument and is not
invented by this implementation. The fixed arm therefore cannot run until the
root-signed corrected bank receipt, schedule artifact, validation metadata,
source commit, and resource lease are supplied.

## Synthetic verification

Command:

```text
PYTHONPATH=.:src pytest -q tests/test_trr_p09_*.py
```

Focused result: **21 passed in 1.44 s**, CPU-only, for the new caller and
shared runner modules. The complete `tests/test_trr_p09_*.py` sweep reached 70
passes and one unrelated pre-existing stage-1 fixture failure: its two-row
synthetic `build_b0_template_buckets` fixture is rejected by the current
120-template invariant. That failure is outside the fixed-control caller and
was not changed. New tests cover metadata join and balanced aggregation,
duplicate/missing-domain rejection, signed grid arithmetic, exact inherited
schedule equivalence, active-prefix/zero-padding validation, create-only
fixed-state serialization with bank binding, state non-mutation, and
curve/exposure/cost receipt serialization.

No production caller, public asset, directional module, or GPU run is claimed
by this checkpoint. Root may wire the fixed hook and the corrected bank once
those signed inputs and live resource guards are available.

# TRR-0011 transfer diagnostic artifacts

`design_v1.md` and `variant_plan_v1.json` are retained as superseded proposal
history. The corrected design is `design_v2.md` with `variant_plan_v2.json` and
`panel_descriptor_template_v2.json`. These artifacts remain design-only until
root and Agent 2 approve an opaque source reservation.

The artificial target families are fixed early and near-cut perturbations at
`1e-3` and `1e-2`, plus one fixed after-cut `1e-2` null control. The historical
`public_lora_2601` condition is recorded as historical trained benchmark metadata and
is unavailable as an independent target. Agent 1 does not duplicate the A1+A2
confirmation owned by Agent 2.

The executable is `scripts/trr0011_transfer.py`. Its validator checks paired
public observations and bound resources before the existing TRR-0010 runner is
loaded; its predictions and directly computed decoder geometry remain
truth-free and create-only. `validate_transfer_prediction_matrix` is a
separate task-local complete-matrix gate for the two fixed methods, all seven declared target conditions, both domains, and 32 records per domain (28 cells);
it does not masquerade as the TRR-0010 128-record gate. CPU synthetic tests are
in `tests/test_trr0011_transfer.py`.

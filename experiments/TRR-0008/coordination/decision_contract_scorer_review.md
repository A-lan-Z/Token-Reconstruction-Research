# TRR-0008 scorer review against the canonical decision contract

The single prospective contract is
`experiments/TRR-0008/planning/decision_contract.json` (schema
`token-reconstruction.trr0008-decision-contract.v1`). The scorer helper
`proposed_decision_contract()` now returns a pointer to that file, which avoids
a second threshold contract. The following bindings still need to be consumed
by the evaluator owner before a frozen score can be used for deployment:

1. `scripts/trr0008_score.py::_decision_parameters` validates the older prose
   fields and hard-codes the margins, route alpha, and CP component alpha. It
   does not validate or read the canonical `confidence`, `cost_gate`, or
   `safeguards.primary_harm` objects. The scorer should fail closed if these
   structured values change and should report them from the loaded contract.

2. `paired_contrast()` hard-codes safeguard CP component alpha `0.025` and
   receives primary component alpha/default bootstrap settings from function
   defaults rather than the loaded contract. The score invocation must bind
   `confidence.primary_quality.exact_cp_component_alpha`,
   `confidence.safeguard.exact_cp_component_alpha`, and
   `bootstrap.{seed,draws,unit}`.

3. The canonical contract declares one-sided bootstrap lower-tail alpha
   `0.025` for the primary token route and `0.05` for safeguard token harm.
   `_bootstrap_interval()` currently constructs central intervals using
   `alpha/2`; therefore `token_net_bootstrap_975` has a `0.0125` lower tail,
   while `token_net_bootstrap_95` has a `0.025` lower tail. `decide()` uses
   the former for primary quality and the latter for safeguards. These outputs
   do not implement the declared one-sided tails. The evaluator must expose
   explicit one-sided lower bounds (record-level bootstrap unit) and use the
   contract's two tail values.

4. The canonical `cost_gate.cells` and `cost_gate.primary_cell` explicitly
   require all four paired cells, with Finance public-base as primary. The
   current timing checker iterates `contract.CELL_ORDER`, which has the same
   membership, but does not validate or report the canonical cost-gate object.
   Bind the timing receipt to the structured cell list and threshold before
   applying the cost gate.

5. The canonical `safeguards.primary_harm` object makes Finance public-base
   exact/token harm a required safeguard as well as a primary quality cell.
   The current loop includes the primary cell indirectly through the safeguard
   list; it should validate this explicit binding and report the primary harm
   result separately.

The scorer already has the corrected Clopper--Pearson tails:
`beta.ppf(alpha_component, gains, n-gains+1)` for the gain lower bound and
`beta.ppf(1-alpha_component, losses+1, n-losses)` for the loss upper bound.
That exact route and its endpoint tests should be retained. This review does
not modify evaluator-owned scorer code.

The numeric paths added to the draft for direct evaluator consumption are
`primary.practical_margin=0.05`, `primary.component_alpha=0.0125`,
`token_endpoint.practical_margin=0.01`,
`safeguards.route_alpha=0.05`,
`safeguards.exact_harm_margin=0.05`,
`safeguards.token_harm_margin=0.01`, and
`bootstrap={seed:8008, draws:10000, unit:"source_record"}`. The cost gate
contains all four cells and binds the PASS precision40 receipt at
`experiments/TRR-0008/timing/precision40_result.json` (SHA-256
`a5d923bb9254f0ba0ec917dc6ede9e22d7b566e47e79408cf188f679c6b30c02`).

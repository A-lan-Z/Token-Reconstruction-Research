# TRR-0005 final report and manifest assembly plan

Status: preparation only. This note defines the post-score assembly procedure;
it does not read the fresh result, evaluator truth, prediction tensors, or
private sidecar. It must be used instead of rerunning the development-only
report builder after the final score exists.

## Authoritative post-score inputs

The final assembly should ingest the following machine-readable files exactly
once after root confirms that the scorer has completed:

- `experiments/TRR-0005/fresh_confirmation_v1/result.json` is the authoritative
  score output.
- `experiments/TRR-0005/fresh_confirmation_v1/freeze_receipt.json` is the
  public gate receipt.
- `experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/panel.json` and
  `observations.json` bind the four cells and their public geometry.
- `experiments/TRR-0005/fresh_confirmation_v1/method_registration.json`,
  `selection_plan.json`, and
  `predictions_v1/{predictions.json,timings.json,run_evidence.json}` bind the
  method matrix and runtime evidence.
- `experiments/TRR-0005/frequency_references_v1.json` supplies both unchanged
  `original` and `enriched` fitting-frequency references.
- The current development manifest and compact receipts under
  `experiments/TRR-0005/{corpus_run,public_activation_v1,joint_fit_v1,joint_fit_qknorm_v1,joint_qualification_v1,joint_qualification_v2,joint_qualification_qknorm_v1,prediction_qualification_v1}/`
  supply historical costs and development findings. Do not rerun
  `build_development_evidence.py` after final scoring because it writes the
  pending development report and manifest.

Before rendering any final claim, fail closed unless the score JSON has
`status == "FRESH_CONFIRMATION_SCORED_AFTER_COMPLETE_PUBLIC_GATE"`,
`truth_gate.verified_before_truth == true`,
`truth_gate.truth_opened_after_gate == true`, exactly four cells, exactly eight
methods, and exactly 32 prediction artifacts and timing receipts. Preserve the
score's receipt, panel, registration, selection-plan, observation, method
binding, and prediction artifact descriptors verbatim in the final manifest;
do not hand-type hashes.

## Score JSON to report mapping

Use `cells_results` as the source for the four separate result tables. Each
entry is keyed as `<cell_id>__<method_id>` and contains `metrics` with scored
and correct token counts, token accuracy, exact-record counts/rate, and the
ordered `per_record` rows. Render Pile/P0, Pile/synthetic-LoRA,
Finance/P0, and Finance/synthetic-LoRA separately. Do not pool domains or
conditions into a headline.

Use `method_comparisons` for paired method contrasts. For each comparison,
retain the baseline/method IDs, token point delta and bootstrap interval,
records with token gains/losses/ties, exact-record gain/loss counts, and both
`exact_beneficial_bound` and `exact_net_benefit_bound`. A positive delta has the
orientation shown by the score JSON's `baseline_method_id` and `method_id`;
do not infer orientation from a label string.

Use `paired_target_comparisons` for P0 versus synthetic-LoRA comparisons. Keep
target pairs clustered by source and report them per domain and method. The
score's `bootstrap` object supplies the shared seed 5005 and 10,000 resamples;
do not regenerate a different schedule.

Use each cell result's `frequency_references.original` and
`frequency_references.enriched` objects for both fitting-frequency views of
every contender. Render the fixed frequency bins, position bins, and joint
frequency × position rows with domain/cell labels. Do not replace the method's
fitting reference with the other map, change bin edges, or pool domains.

The final report's primary comparison block should distinguish the following
predeclared labels:

- `enriched__causal_vs_diagonal` and
  `enriched__causal_vs_best_positionwise` are the extra-H contrasts. The latter
  uses the frozen public affine-versus-diagonal selection only; it never
  selects against causal. Since both distributions selected trained diagonal,
  these two enriched contrasts are duplicate method pairs and are not
  independent corroboration.
- `coverage__<state>__enriched_vs_original` is the enrichment comparison for
  each decoder state; report it for each cell/target and never as a pooled
  domain claim.
- The anchor labels compare the retained A1/A2 methods with the original
  affine baseline and should remain a separate runtime/accuracy context.

Apply the already frozen practical margins from the decision plan: extra-H
uses 0.5 percentage points for token accuracy and 5 percentage points for
exact rate; enrichment uses 2 percentage points for token accuracy and 5
percentage points for exact rate. Report the score's finite-sample bound and
margin booleans alongside every relevant endpoint. Zero discordances retain a
positive finite-sample upper bound (about 4.93 pp at n=128 under the declared
family tail), so they cannot be described as equivalence. Use the root-approved
decision script/formula for the final pass/fail wording rather than inventing a
new rule during rendering.

## Cost and footprint scope

Keep these buckets separate in the report and manifest:

| Bucket | Scope to ingest | Accounting rule |
|---|---|---|
| Corpus preparation | child/launch timing, RSS, 11.474M extra public occurrences scanned, coverage IDs, controlled placements at token positions >=128 | report preparation and scanning once; no holdout source contents |
| Public activation capture | authoritative capture and launch timing, batch-8 x 192 bit-exact path, GPU/host peaks, excluded batch-1 non-equivalence | retain the excluded diagnostic as excluded evidence |
| Decoder fitting | six original fits plus two qknorm causal fits, 3,000 steps, 1.536M draws/arm, about 801.7 s aggregate fit wall time | the two original dot-causal fits are successful completed development fits superseded by the preregistered qknorm repair; keep them as development context, not failed attempts |
| Public diagnostics | old and qknorm H-only attention runs | no truth or E-table claim; distinguish tested score branches from general earlier-H usefulness |
| Qualification | preserved V1 failure, V2 largest-cell qualifier, qknorm qualifier, and archived Finance-128 prediction qualifier | report each receipt and failure separately |
| Final inference | `predictions_v1/run_evidence.json` plus per-cell `*.run.json` | count `runtime_load_seconds` once per method; retain warm CPU-H-to-CPU-ID and measured-call simulation counts and peak scopes; do not multiply repeated load fields or make near-identical speed claims |
| Static assets | shared E 1,050,673,488 bytes; learned state files approximately 17/21 MB; A2-only P0 weights 2,471,645,608 bytes plus config | report role and byte count from registered descriptors; historical A1 preparation cost is unknown unless a receipt supplies it |

The final manifest should retain the actual result file descriptor, freeze
receipt descriptor, prediction/timing/registration/panel/observation bindings,
frequency-manifest descriptor, and external evaluator receipt only after they
are machine-ingested. Retaining the largest raw H/fit tensors in the task
archive is a packaging choice; record their paths, bytes, hashes, and replay
commands when archived. The truth sidecar remains outside the frozen
prediction root at scoring time.

## Final report order and limitations

Lead the completed report with the four-cell fresh findings: method/target
counts, token and exact outcomes, primary paired contrasts, and uncertainty.
Follow with frequency × position × domain diagnostics, gains/regressions, and
then the development context and cost accounting. State that all 32
method-cell artifacts passed the public gate before truth opened, if and only if
that is present in the score receipt.

Retain these limitations in the final interpretation:

- The fresh panel has 128 paired sources per domain and two target conditions;
  target pairs share sources and are not independent observations.
- Public positionwise selection was affine versus diagonal only and was frozen
  before fresh evaluation. Both selections being diagonal makes the two causal
  enriched contrasts duplicated, not independent evidence.
- The old dot-product attention result is a tested-branch routing diagnostic;
  qknorm's improved earlier-position mass does not establish that earlier H is
  necessary or generally useful.
- Trained diagonal retains the contextual current vector H_i and adds a
  positionwise nonlinear correction through the declared layer-normalized
  value path; it is not context-free or merely a redundant affine map. Causal
  adds earlier H through the same added path, and qknorm is a routing repair;
  disclose the inactive Q/K degrees of freedom under the one-key diagonal
  mask while retaining the declared parameter footprint.
- Original and enriched frequency maps are both descriptive views of the same
  predictions; migration across maps is not a new evaluation sample.
- Retained fit-bank diagnostics (original 99.9935676% / 1,192 of 1,200 exact;
  enriched 90.4447178% / 3 of 1,200 exact; initial identity approximately 31%)
  describe development initialization. The two original dot-causal fits are
  successful completed development fits superseded by the preregistered
  qknorm repair; final fit-stream 100% values do not imply that every earlier
  selected state was perfect. The actual failed attempts are the preserved V1
  qualification forecast guard and the capture output-root collision.

The final manifest status and access flags may be advanced only after the
machine score passes the checks above: set fresh evaluation complete, holdout
selected, and truth opened according to the actual receipts. Keep the current
pending report and manifest unchanged until then.

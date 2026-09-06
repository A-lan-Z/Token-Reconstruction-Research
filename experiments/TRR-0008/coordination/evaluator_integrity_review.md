# TRR-0008 evaluator integrity review

Review scope: current TRR-0008 capture, registration, runner, gate, truth,
selector, and scorer drafts, checked against `RESEARCH_CHARTER.md`, the
TRR-0008 evaluation interface, and the current pre-truth planning contract.
This note contains metadata/code-path findings only; no source text, target
labels, truth sidecar, fresh selection, or unpublished parallel study output
was opened. Evaluator-owned source was not modified.

## Findings for owner repair

1. **High — scorer can read an unfrozen prediction root.**
   `scripts/trr0008_score.py::_load_predictions` (around lines 577–584) builds
   paths from the user-supplied `--predictions-root` and only opens the
   `predictions` tensor key. `main` never checks that this root is the
   registered `output_root`, the freeze receipt's run-manifest root, or the
   artifact bindings returned by `gate.validate_before_truth`. A complete
   public gate can therefore be followed by scoring a different directory.
   Bind the scorer's prediction root to registration/freeze output and run the
   contract artifact/hash validation on the exact files that will be scored.

2. **High — scorer does not bind the truth payload to its recorded digest.**
   `scripts/trr0008_score.py` checks only that `--truth` has the same path as
   the metadata-only header (around lines 620–632), then opens it. It does not
   compare the sidecar's current byte count/SHA-256 to `header.sidecar`, nor
   validate the sidecar metadata fields for registration, source selection,
   observation record digests, and domain counts. A replacement at the same
   path can be scored after the pre-truth gate. After authorization to open
   truth, verify the bound file record and the required metadata before reading
   tensors.

3. **High — an alternate timing receipt can reach the decision.**
   With `--timing-receipt`, scorer `main` (around lines 636–647) uses that path
   for `decide()` without checking it against the freeze receipt's timing
   binding. The pre-truth gate validates the receipt-bound timing path, but the
   later override can change the cost decision. Require the supplied path and
   file record to equal the frozen receipt, or remove the override at scoring.

4. **Medium — truth header cell digests are presence-only at the second gate.**
   `scripts/trr0008_eval_gate.py::validate_before_truth` (around lines
   295–310) checks that every cell has a string `record_ids_sha256`, but does
   not compare those values to the current registered observation manifest and
   public freeze receipt. Require four unique cells and exact digest equality
   before the sidecar is opened. `eval_truth.py` performs the correct
   selection/observation comparison while preparing truth, but the later gate
   must protect the scorer against header or observation replacement.

5. **Medium — resource checks fail open when host telemetry is unavailable.**
   `scripts/trr0008_eval_runner.py::_guard` (around lines 111–127) skips RSS
   and host-memory thresholds when `_rss_bytes()` or `_host_available_bytes()`
   returns `None`. The active evaluator path therefore continues without a
   fail-closed resource guard on telemetry failure. Raise a runner error for
   missing required telemetry. The runner also lacks the timing helper's
   foreign-GPU-process check, so CUDA execution can share a device without a
   recorded exclusivity failure.

6. **Medium — numerical inter-op setting can silently differ.**
   `eval_runner._configure_numerics` (around lines 90–108) catches the
   `set_num_interop_threads` runtime error and continues. It records the
   effective value but does not reject a value different from the registered
   32. Treat a mismatch as failed closed before loading models.

7. **Medium — registration CLI drops timing bindings.**
   `scripts/trr0008_eval_register.py::main` (around lines 285–295) parses
   `--timing-plan` and `--timing-receipt` but passes only `plan_path` to
   `build_registration`. A CLI-created registration can silently omit the
   canonical timing plan/receipt even when supplied. Pass both parsed options
   and require their bindings when the final cost gate is frozen.

8. **Medium — runner failure leaves a partial create-only matrix without a
   failure receipt.**
   `eval_runner.execute` writes `registration.json` and per-cell artifacts as
   it proceeds, but `main` only prints an error. A model/load/resource failure
   can leave an unusable partial output that blocks restart and lacks a
   structured failure artifact. Preserve a failed-closed receipt with stage,
   resource checks, and partial paths, or make the run directory transactionally
   invalid until the complete matrix gate passes.

9. **Low — optional code bindings can omit required files.**
   `eval_register.py::build_registration` (around lines 199–213) silently skips
   a missing code-binding path. The registration contract accepts the resulting
   shorter list. Require the complete declared binding set for a final freeze;
   this is separate from the TRR7 timing diagnostic's already-checked bindings.

## Source-free runner qualifier prepared

`scripts/trr0008_eval_qualifier.py` uses the proven TRR7 registration,
observation, state, embedding, and archive loaders, but invokes the actual
`scripts.trr0008_eval_runner.predict_current_h` for the comparison. It checks
the four scientific methods × four public cells × 128 rows, exact tensor
equality, fixed fail-closed resource checks, and no-truth/source flags. It
captures the executable commit and qualifier/runner/contract/loader bindings
before loading and rechecks them before success serialization. It writes only
a create-only digest receipt or a separate failure receipt:

- success: `experiments/TRR-0008/evaluation/qualifier/runner_qualifier.json`
- failure: `experiments/TRR-0008/evaluation/qualifier/runner_qualifier.failure.json`

The fixed command and no-retry rule are in
`experiments/TRR-0008/evaluation/qualifier/qualifier_plan.md`. CPU fixture
tests exercise the actual runner adapter across all 16 matrix entries and a
single-row corruption failure:

`PYTHONPATH=.:src:scripts pytest -q tests/test_trr0008_eval_qualifier.py`

Result at preparation time: **4 passed**. The CUDA qualifier has not been
executed; it awaits root commit and explicit compute authorization.

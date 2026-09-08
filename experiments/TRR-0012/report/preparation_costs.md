# TRR-0012 preparation cost ledger

This report is derived from existing JSON/JSONL receipts only. It does not include substantive B0/B1 fitting or evaluation. Outer watchdog/process walls and inner child/phase walls are shown separately; nested timings are never added.

The listed distinct external invocations sum to **250.496860 seconds**. That number includes failed guarded attempts, excludes metadata-only attempts without recorded duration, and excludes unknown historical source/ancestor preparation. It is not an end-to-end wall clock.

## External attempt ledger

| Attempt | Status | External wall (s) | Peak host RSS | Disposition |
|---|---|---:|---:|---|
| `b1_cpu_r2` | COMPLETED_SUPERSEDED_EXCLUDED | 39.225268 | 1,623,183,360 B (1.512 GiB) | Retained CPU input compilation; excluded from final capture. |
| `b1_cpu_r3_source26e_initial` | EXCLUDED_FAILED_BEFORE_INPUT_RECEIPT | 2.023494 | 544,911,360 B (0.507 GiB) | Preserved watchdog failure; no preparation manifest was produced. |
| `b1_cpu_r3_source26e_retry` | EXCLUDED_FAILED_BEFORE_INPUT_RECEIPT | 3.028907 | 560,779,264 B (0.522 GiB) | Preserved watchdog failure; no preparation manifest was produced. |
| `b1_cpu_r3_source26e_final` | COMPLETED_SUPERSEDED_EXCLUDED | 36.209942 | 1,607,741,440 B (1.497 GiB) | Retained source26e CPU input compilation; excluded from final capture. |
| `b1_cpu_r4_date07` | COMPLETED_CONSUMED_BY_CAPTURE | 33.195524 | 1,651,253,248 B (1.538 GiB) | Final historical-date-compatible CPU input compilation used by capture. |
| `public_validation_r1` | PASS_PUBLIC_VALIDATION_PREPARED_NO_TRUTH | 6.045456 | 1,159,860,224 B (1.080 GiB) | Completed public validation preparation; no public forward was recorded. |
| `schedule_b0_r1` | PASS_CPU_B0_SCHEDULE_PREPARED_NO_MODEL | 7.049109 | 810,188,800 B (0.755 GiB) | Completed B0 schedule preparation; superseded by the final common schedules r4 receipt. |
| `schedules_r4_common` | PASS_CPU_COMMON_SCHEDULES_PREPARED | 11.069470 | 859,021,312 B (0.800 GiB) | Final common B0/B1 schedules used by native configuration and fixed qualification. |
| `capture_qualification_r1_import_path` | EXCLUDED_METADATA_PREFLIGHT_FAILURE | — | — | Trusted public loader import omitted the historical source scripts root; stopped before input validation and before any model or CUDA use. |
| `capture_qualification_r1_manifest_absolute_path` | EXCLUDED_METADATA_PREFLIGHT_FAILURE | — | — | The hash-bound preparation manifest retains an absolute /tmp source-root dependency; running from the durable archive made the signed relative addendum resolve to a different path. |
| `capture_qualification_r2_scratch_source_closure` | PASS | — | — | Exact 24a files and all signed dependencies hash-match the durable snapshot; metadata-only plan, input, model, watchdog, and authorization validation passed before guarded execution. |
| `capture_qualification_r2_pass` | QUALIFICATION_PASS | 8.098524 | 2,444,636,160 B (2.277 GiB) | Completed native public-base qualification; no activation capture, fit, evaluation, or truth access. |
| `full_capture_b1_date07` | CAPTURE_COMPLETE_NO_TRUTH | 58.365058 | 1,847,615,488 B (1.721 GiB) | Completed 10,800 new-row public B1 activation capture and integrity audit. |
| `configuration_dry_run_r1_B1` | EXCLUDED_FAILED_CLOSED | 3.050789 | 659,787,776 B (0.614 GiB) | Preserved failed native B1 configuration attempt; r2 repaired the immutable B0 binding path and passed both banks. |
| `configuration_dry_run_r2_B0` | PASS_FIXED_CONTROL_CPU_CONFIGURATION | 7.096066 | 621,076,480 B (0.578 GiB) | Completed native B0 configuration-only validation; no model, E, optimizer, forward training, GPU, or evaluation truth. |
| `configuration_dry_run_r2_B1` | PASS_FIXED_CONTROL_CPU_CONFIGURATION | 13.655427 | 635,023,360 B (0.591 GiB) | Completed native B1 configuration-only validation; no model, E, optimizer, forward training, GPU, or evaluation truth. |
| `fixed_qualifier_v1_source_binding` | EXCLUDED_FAILED_BEFORE_PREFLIGHT | — | — | Preserved source-binding failure; no preflight, model, tensor, or GPU access. |
| `fixed_qualifier_v1_watchdog_import` | EXCLUDED_FAILED_BEFORE_CHILD | 0.126433 | — | Fresh preflight passed, but the direct watchdog process exited before creating watchdog or child receipts. |
| `fixed_qualifier_v2_pass` | QUALIFICATION_PASS | 22.257393 | 3,415,343,104 B (3.181 GiB) | Completed one bounded native B1 first-eight-update probe; all updates and selected state were discarded; no checkpoint or contender was retained. |

## Nested phase observations

- Public validation preparation receipt inner elapsed: `4.252861926` s; external watchdog: `6.045456` s.
- Capture qualification inner elapsed: `0.682464084` s; external watchdog: `8.098524` s.
- Full capture child wall: `51.255044390` s; external watchdog: `58.365058` s. Recorded inner phases are `{"forward": 30.631023660007486, "input_validation": 0.0, "resume_verify": 0.008757938003327581, "write": 19.617353111007105}` s.
- Fixed qualifier v2 child wall: `20.717105252` s; runner whole wall: `5.378993671` s; external watchdog: `22.257393` s.

The fixed-qualifier peak GPU reserved value is **2,959,081,472 bytes = 2.755859 GiB (2.756 GiB rounded)**. The byte value is retained in the JSON ledger with the other resource observations.

## Final and excluded public-input decisions

- `b1_cpu_r2` completed but is excluded because the current-date Alpaca rendering changed 5,400 rendered/public hashes.
- The source26e preparation completed with the same current-date mismatch; its two earlier failures are preserved separately: missing `scripts.trr0007_support_diagnostics`, then unavailable approved exclusion ledger.
- `b1_cpu_r4_date07` pins the historical `07 Sep 2026` Alpaca date and is the input consumed by capture.
- Capture qualification metadata-only attempts `r1_import_path` and `r1_manifest_absolute_path` are excluded without recorded durations; `r2_scratch_source_closure` passed.
- Configuration dry-run r1 B1 failed closed on immutable B0 binding validation; r2 B0/B1 both passed.
- Fixed qualifier v1 first failed source-binding freeze before preflight; its next launch failed before child start on watchdog import. Fixed qualifier v2 passed with eight discarded updates and no retained checkpoint or contender.

## Historical or unknown preparation

Source snapshot creation/restoration, the TRR-0009 starting decoder ancestor, and the TRR-0003 public normalized embedding are reported separately as historical or unknown because no TRR-0012 preparation timing receipt prices them. Their exact binding paths and hashes are in `preparation_costs.json`.

## Key receipt hashes

- `final date07 preparation manifest`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/experiments/TRR-0012/preparation/b1_cpu_r4_date07/preparation_manifest.json`; SHA-256 `d1097b467e4abe4ec9f39dfca27f34f1d481527d2de79ca610751a8fe260620e` (42163 B).
- `public validation preparation receipt`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/experiments/TRR-0012/preparation/public_validation_r1/preparation_receipt.json`; SHA-256 `20b3cdad32ffc4117529217e3cf24abace0becc39c0741786dbef668f23b479e` (6496 B).
- `final common schedules receipt`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/experiments/TRR-0012/preparation/schedules_r4/schedule-receipt.json`; SHA-256 `87dc642ee7e883a6282a1111c74df31e948ebe3f6abf1e921725f15e22e0b583` (5983 B).
- `capture qualification receipt`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/experiments/TRR-0012/capture/qualification_date07_r2/qualification-receipt.json`; SHA-256 `1e897a0eee1ec6e51368bc01e783586b3e29ceab0c6b597ad64d1444c3494bf5` (7420 B).
- `full capture receipt`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/experiments/TRR-0012/capture/b1_activation_date07/capture-receipt.json`; SHA-256 `2980d5115b6cf2a1d1c910fdf37da47e8f237d31f2b643af9fad526462532766` (10778 B).
- `full capture completion receipt`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/experiments/TRR-0012/capture/capture_completion_v1.json`; SHA-256 `ef870fd2c2382dbab05c291e8ce5af86a820e809ffbab7ac3348cc0a09d0a77e` (9826 B).
- `native configuration receipt`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/experiments/TRR-0012/execution/configuration_dry_run_r2/configuration_dry_run_receipt.json`; SHA-256 `caa7b6ce551e2e3d9a5bce6652da620a2d4a4117a64d8b76be60ca3b1c4ab47d` (5840 B).
- `fixed qualifier v2 PASS receipt`: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/outputs/TRR-0012/qualification_fixed_b1_v2/qualification_receipt.json`; SHA-256 `2eba2b75b64015911f9801637814e5de9698a2f00824f67980d720a14872c706` (19925 B).

The JSON companion contains the complete evidence-path/hash inventory, per-attempt resource values, recorded exclusion reasons, and the unaggregated inner timing observations.

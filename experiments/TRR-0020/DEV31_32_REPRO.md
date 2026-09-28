# Development31/32 reproduction and evidence

All work is local on task/TRR-0020. No public push or PR was attempted because
the previously recorded automatic publication approval rejection persists.
The complete no-shortlist reconstruction objective remains unmet unless later
paired canonical evidence establishes it.

Environment: Linux Python3.12, PyTorch2.10.0+cu128, Transformers5.3.0, RTX5080.
Each probe/phase records the exact environment and full execution commit.
Use PYTHONPATH=src, CUBLAS_WORKSPACE_CONFIG=:4096:8,
PYTHONDONTWRITEBYTECODE=1; TF32 disabled and deterministic algorithms enabled.
Outputs/receipts are create-only. Use a separate checkout/output tree for replay;
do not remove or overwrite preserved evidence.

Public fixture: seed200051, token IDs generated solely to construct synthetic H,
lengths128/40. No target-label file is used by public probes. The development
matrix uses the unchanged eight records from TRR-0014/fresh_r1 and is retrospective.
Reconstruction code receives observations/metadata; grid_score opens labels only
after complete56-cell, source/asset, output, numerical-control and negative gates.

Relevant execution commits:
- 70685dd: original development31 public probe; preserved excluded capture failure.
- 4920153: development31 r1 probe; exact reproduction of all38completed old artifacts.
- 27151cc: development32 public budget/iteration comparison.
- f1006b0: development32 reconstruction grid.

CPU commands:
python3 scripts/trr0020_stage31/cpu_reference.py
python3 scripts/trr0020_stage32/cpu_check.py

The CPU JSON receipts include exact source hashes, command, timestamps, result,
stderr and exit status. Their command stdout is the numerical result object;
the recorded execution wrapper adds provenance.

Guarded public commands (from the worktree root):
python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev31_guard.json --timeout 600 -- python3 scripts/trr0020_stage31/probe.py
python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev31_guard_r1.json --timeout 600 -- python3 scripts/trr0020_stage31/probe_r1.py
python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev32_guard.json --timeout 600 -- python3 scripts/trr0020_stage32/probe.py

Development31's first command is a preserved failure, not the accepted execution.
DEV31_REPAIR.md explains the autograd lifetime repair. The r1 command requires
restored failed-attempt artifacts as exact reproduction anchors. Public probes
also require historical warm/mixture controls; their receipts identify all paths
and hashes. Restore these from their portable archives when absent.

Reconstruction and scoring:
python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev32_grid_guard.json --timeout 1500 -- python3 scripts/trr0020_stage32/grid_run.py
python3 scripts/trr0020_stage32/grid_score.py

The grid launches one isolated restart-safe worker per frozen configuration.
Restart the same grid only after its process is confirmed terminal; it verifies
completed receipts and refuses changed/orphaned outputs. A scorer failure must
not be bypassed to open labels. Qualification includes full-logit update controls,
public first-step anchors, repeated/eager decode equivalence, independent gradients
and direct FP64 KL at declared public steps.

Archive commands:
python3 scripts/trr0020_stage31_archive.py
python3 scripts/trr0020_stage32_archive.py
python3 scripts/trr0020_stage32_grid_archive.py
python3 scripts/trr0020_stage31_summary.py
python3 scripts/trr0020_stage32_grid_summary.py

Portable evidence uses sha256-object-archive-v1. Concatenate parts in listed order,
verify the archive hash, and restore each logical path from its mapped ZIP object.
Every member was round-trip verified. Development31 includes the excluded attempt.
dev31_archive_metadata_correction.json corrects an old hardcoded command label;
its raw artifact hashes remain unchanged.

Timing: public rows report eager probability-update time, separate from validation
and the model gradient. Grid total includes the actual64-update decoder and
diagnostic transfers; prefix/table preparation and graph captures are recorded
separately. Checkpoint times include the update immediately after the corresponding
state evaluation and are not minimal standalone early-stop implementations.
The four/eight/sixteen scalar solvers are distinct numerical rules, not equivalent
execution ports. No canonical replacement claim follows from this development grid.

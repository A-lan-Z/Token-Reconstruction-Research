# Continuation19 reproduction and provenance

All work is on local branch task/TRR-0020. Public publication remains subject to the previously recorded automatic approval-review restriction.
The user requirement remains full-vocabulary reconstruction without a proposer or a separate candidate verifier. No successful replacement is declared.

Run from this worktree with PYTHONPATH=src, CUBLAS_WORKSPACE_CONFIG=:4096:8 and PYTHONDONTWRITEBYTECODE=1. Do not rerun create-only paths over preserved evidence; use an isolated checkout/output namespace and retain hashes.

## Frozen CPU checks

- experiments/TRR-0020/dev40_cpu_reference_failed01.json: passed=False; commit `6802969dd3c6e6f9dc8ff67a401afa4d43116c3f`; command `["/usr/bin/python3", "scripts/trr0020_stage40/cpu_check.py"]`.
- experiments/TRR-0020/dev40_cpu_reference.json: passed=True; commit `51065d5f90c3cb86f5e71e29fb843af50a319321`; command `["/usr/bin/python3", "scripts/trr0020_stage40/cpu_check.py"]`.
- experiments/TRR-0020/dev42_cpu_reference.json: passed=True; commit `9017aa5aa4bf2320a790a90e044b8661be445cca`; command `["/usr/bin/python3", "scripts/trr0020_stage42/cpu_check.py"]`.

## Exact guarded GPU launches

- experiments/TRR-0020/dev39_gpu_launch.json: commit `9add847a945744ea0e91e6ebf7cace52ae9ec43f`.
  Command: `["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev39_gpu_guard.json", "--timeout", "900", "--", "python3", "scripts/trr0020_stage39/gpu_qualification.py"]`.
- experiments/TRR-0020/dev40_gpu_launch.json: commit `b7af7685917bf0e528e191307eb229456ab45dec`.
  Command: `["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev40_gpu_guard.json", "--timeout", "600", "--", "python3", "scripts/trr0020_stage40/gpu_audit.py"]`.
- experiments/TRR-0020/dev41_profile_launch.json: commit `c4e02a9c7a7c78fd54abe652e3e0361ce953745a`.
  Command: `["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev41_profile_guard.json", "--timeout", "600", "--", "python3", "scripts/trr0020_stage41/profile_current.py"]`.
- experiments/TRR-0020/dev41_profile_r1_launch.json: commit `2a3b31ab7c6ae84c7ff4deea1bb30630ce6271a1`.
  Command: `["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev41_profile_r1_guard.json", "--timeout", "600", "--", "python3", "scripts/trr0020_stage41r1/profile_current.py"]`.
- experiments/TRR-0020/dev41_profile_r2_launch.json: commit `03c984aaf7bd6592f07b588acebf9797c7aaa4fc`.
  Command: `["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev41_profile_r2_guard.json", "--timeout", "600", "--", "python3", "scripts/trr0020_stage41r2/run.py"]`.
- experiments/TRR-0020/dev42_gpu_launch.json: commit `fdcd7776f1d8da3cb2f72c9489e674a59a9d6002`.
  Command: `["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev42_gpu_guard.json", "--timeout", "600", "--", "python3", "scripts/trr0020_stage42/gpu_check.py"]`.
- experiments/TRR-0020/dev43_launch.json: commit `63f1b1d13f99aae108aa3186ea0454afef158455`.
  Command: `["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev43_guard.json", "--timeout", "1200", "--", "python3", "scripts/trr0020_stage43/run.py"]`.

Development39 was run before this continuation; its completed evidence was summarized here. Development40 diagnosed the inadequate inverse and does not instantiate a decoder.
Both early development41 profiles failed at second-geometry stream capture. Their partial results remain excluded and archived. Revision2 uses detached fixtures and separate sequential geometry processes; all original outputs/traces reproduce.
Development42 changes floating-point association, preserving the mathematical full-vocabulary update. Its isolated-update speed is not end-to-end inference speed.
Development43 freezes every original/fused64/128-step development cell before retrospective scoring. No source token labels enter generation. Canonical replacement requires registering any advanced contender and completing both canonical setups with the entire active matrix.

Archives use exact-byte hash deduplication. Concatenate numbered parts, check archive_sha256, then map logical_members to objects. Raw external files, launch receipts, environment records, prefix/fixture hashes and phase timings remain in their manifests.


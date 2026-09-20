# TRR-0020 continuation22 and final benchmark reproduction

The owner's latest instruction is preserved verbatim in
coordination/requests/TRR-0020-STOP-AFTER-CURRENT-BENCHMARK.md.
Finish the already registered canonical moment comparison, document it, and
stop. These commands describe the record; they are not a queue for further work.

Use the Linux worktree at
/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0020.
The desktop runs on Windows; user-facing file links use its Ubuntu WSL UNC path.

## Numerical method

The final registered contender is no_shortlist_bennett64_positionbest:
unchanged development50 MomentOptimizer, mode bennett,64updates,
best_position_error. Each update represents all128256vocabulary entries.
The prefix is frozen. The method fits no additional model parameters and uses
no top-K shortlist or separate candidate verification.
It still builds an initialization table from prefix weights.

Development49 R1 qualifies the moment-bound step against independent CPU
references and fixed public GPU states. Development50 compares complete
reconstructions at16/32/64/128updates. The chosen64-update readout was selected
before canonical execution. Changing that rule after scoring would be a new
experiment, which the owner has not authorized.

## Execution environments and dependency layout

The recorded environment is Python3.12.3, PyTorch2.10.0+cu128,
Transformers5.3.0, Triton3.6.0 and an RTX5080 with16303MiB.
Each launch/phase receipt retains the exact observed environment and full commit.

Novel-method workers use PYTHONPATH=src,
CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTHONDONTWRITEBYTECODE=1,
deterministic algorithms and TF32disabled.
A1+A2 workers intentionally unset CUBLAS_WORKSPACE_CONFIG and disable
deterministic algorithms while retaining TF32disabled, matching their archived
execution. The coordinator sets these environments separately.

The original canonical_moment matrix failed before numerical work because its
shared import required the novel environment inside the baseline process.
The original source,52successful qualification checks, launch and guard remain.
R1 imports the novel module lazily and verifies its binding in an isolated,
CPU-only subprocess for baseline workers. Both environment-isolation checks
passed without CUDA initialization.

Required external inputs are hash-bound in development_binding.json and each
prediction freeze. They reside in sibling worktrees:
- agent4-prefix-only-inversion/outputs/agent4-prefix-only-inversion/backup:
  supplied public prefix, configuration and existing public Alpaca A1 lens;
- TRR-0014/outputs/TRR-0014/fresh_r1:
  eight previously opened development observations;
- TRR-0014/outputs/TRR-0014/lookup_dev_r1_responses.safetensors:
  inherited hash-bound table used by earlier code dependencies;
- TRR-0018/outputs/TRR-0018:
  canonical metadata/observations, archived A1+A2 predictions and evaluator truth.

Only scoring opens evaluator_truth.safetensors after the complete current freeze.
Canonical records were opened historically, so this is retrospective evidence.

## Commands and immutable output namespaces

Each exact command, environment, full code commit, start/end time and return
code is retained in its launch receipt and in CONTINUATION22_PROVENANCE.json.
The principal scripts are:

1. scripts/trr0020_stage49_r1/moment_cpu_reference.py
2. scripts/trr0020_stage49_r1/gpu_check.py, under trr0020_resource_guard.py
3. scripts/trr0020_stage50/run.py, under the resource guard
4. scripts/trr0020_stage50/score.py
5. scripts/trr0020_stage49_archive.py and trr0020_stage50_archive.py
6. scripts/trr0020_canonical_moment_r1_import_check.py
7. scripts/trr0020_canonical_moment_r1/validate.py, guard timeout600s
8. scripts/trr0020_canonical_moment_r1/run.py, guard timeout3600s
9. scripts/trr0020_canonical_moment_r1/score.py
10. scripts/trr0020_canonical_moment_r1_archive.py
11. scripts/trr0020_canonical_moment_r1_summary.py
12. scripts/trr0020_closeout_ledger.py

For the canonical run, require at least10GiB GPU free and no competing GPU
processes at launch. Qualify all52checks before releasing the full matrix.
The matrix has192inputs ×3methods ×3repetitions =1728replicate cells, with method
order rotated between repetitions. The first method in each isolated worker
also undergoes three largest-case decodes. Runtime guard and peak-memory limits
are preserved in preflight.json and BENCHMARK_PLAN.md.

Canonical source commit:
649abcbd69d64ece05399e12b0d26412137a6286.
R1qualification source commit:
30c6b0fe10a7f5d478cbac7da4f13718e4b1aced.
Development50 inference and scoring commit:
feb90eb8a66a824ca867dccce2ab1e99630f2fc1.
Development49 R1 GPU commit:
6398cfff8d03f324ab10971b6ad5a2cd5ca8b25e.
Development49 R1 CPU commit:
a38d644fb3ceab3f240030e6c0b505c2297c1c52.

Outputs are create-only. Do not rerun into an occupied namespace or delete
evidence. A full reproduction requires a separate checkout at the recorded
commit with the same relative sibling layout and restored predecessor artifacts.
Do not edit frozen source, plan, registry or bindings to bypass their checks.

## Restoring and checking evidence

Each archive manifest maps logical paths to ZIP members and records hashes.
For split archives, concatenate part files in the listed order, verify the
whole archive SHA256, then verify each recovered member against its recorded
hash. Deduplicated archives store several logical paths as one SHA256 object.
Keep original path mappings when restoring.

The current archive includes all576unique prediction files, all1728cell
receipts, nine phase receipts,27largest-case raw runs and both original/R1
52-check qualification sets, with their receipts. The original import failure
is retained. Prior study archives remain referenced rather than recopied.

RESEARCH_LEDGER.json indexes all50research stages, the36accepted reconstruction
matrices/2816method-input cells, source files, failures and canonical evidence.
It hashes source and metadata files and points to existing raw archive manifests;
it does not claim to rerun or reverify every historical binary archive.

The readable consolidation is coordination/results/TRR-0020-CONSOLIDATED.md.
The detailed chronological report remains coordination/results/TRR-0020.md.

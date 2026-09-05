# TRR-P02 run1 failure and retry validation

This receipt records the first leased attempt and the minimal isolated-runner
fix. It is separate from the scientific result and contains no public geometry
measurements.

## Failed attempt

- implementation commit: `3110ea7fa1979e6fcaa494f7933c002d51d3c20d`
- lease start/end: `2026-09-05T11:59:00Z` / `2026-09-05T12:11:00Z`
- actual process start/end: `2026-09-05T12:00:03Z` / `2026-09-05T12:00:08Z`
- output root: `experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1`
- exit status: `1`; timeout: `false`
- outer Max RSS: `2,431,792 kB`; swaps: `0`
- preserved evidence: `preflight.json`, `outer-time-v.txt`,
  `outer-stdout.txt`, `outer-stderr.txt`, `outer-receipt.txt`, and
  `failure-receipt.json` in the run1 directory

The exact command was the approved invocation below, wrapped by
`/usr/bin/time -v` and executed with stdout/stderr redirected outside the run
root.

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /usr/bin/time -v -o /tmp/trr-p02-time-20260905-run1.txt timeout --foreground 660s python3 scripts/trr_p02/diagnose_geometry.py --plan experiments/TRR-P02/plan.json --prototype /tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors --historical-lens /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --output-root experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1 --manifest-path experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run1/run_manifest.json --implementation-commit 3110ea7fa1979e6fcaa494f7933c002d51d3c20d --prototype-chunk-size 8192
```

The command passed resource preflight, loaded the P01 table and public model
(the 146/146 weight shards completed), then failed while importing the
historical-lens implementation:

```text
ModuleNotFoundError: No module named 'reference'
HistoricalComparatorError: published strict-BOS lens loader is unavailable
```

The lens checkpoint was not loaded. Qualification, public-prefix forwards
after model load, panel collection, geometry summaries, rankings, and all
scientific outputs were not started or produced. This is a failed dependency
wiring attempt, not a zero-model-load claim.

## Minimal fix and model-free validation

Fix commit: `c4edb0f8de25c1897f8e4f7241a432cfe3db9923`. The runner now inserts
its project root (`_SOURCE_ROOT`) into `sys.path` before importing the original
`reference.strict_bos.round001_teacher` module. No replacement lens or copied
implementation was introduced.

Exact import validation after the fix:

```text
PYTHONPATH=src:scripts/trr_p02 python3 - <<'PY'
import diagnose_geometry
from reference.strict_bos.round001_teacher import FrozenAffineLens
print('source-path/import smoke: PASS', FrozenAffineLens.__name__)
PY
source-path/import smoke: PASS FrozenAffineLens
```

The focused model-free suite was also run after the one-line fix:

```text
PYTHONPATH=src:scripts/trr_p02 pytest -q tests/test_trr_p02_geometry.py
.......                                                                  [100%]
7 passed in 0.69s
```

No retry was launched under the released lease. A retry requires a new explicit
UTC lease, a fresh run2 output root and manifest, the new committed SHA, and the
same CPU-only/resource command.
## Run2 and run3 failed attempts

The run2 and run3 attempts are preserved separately under
`experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run2` and
`experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run3`. Their
`failure-receipt.json` files retain the exact command, phase, timestamps,
resource receipt, and artifact hashes.

- run2 (`da818f7a08f70a04a93fc1d627812925b5d42fb1`): public model, frozen
  lens, qualification, panel/cache checks, and rankings completed; figure
  generation failed before serialization because seven context labels were
  paired with five primary offset rows.
- run3 (`b0d8e03d0fab12842d64d5bc1ac72d1a4d4580cd`): the plotting fix allowed
  both figures and all numerical phases to complete; safetensors rejected the
  shared storage of `primary_activations` and the `baseline` view used for
  `recomputed_baseline`. No diagnostics or ranks were retained from this
  failed serialization attempt. Receipt SHA256:
  `308052ed547a4266671d6172f4c5a4fdbaef4c475465b6b6432743dc630106aa`.

## Serialization fix validation

The one-line source fix in commit
`470b6f1becfaa6da110048302938feddd7204c30` changes only
`recomputed_baseline` serialization to
`baseline.detach().clone().contiguous()`. The source compiles, and a
model-free safetensors smoke built the same view/clone relationship as the
runner, checked unique storage pointers for all 14 saved keys, saved and
reloaded the artifact, and verified the baseline values and int32 neighbor IDs:

```text
serialization-smoke: PASS
py_compile: PASS
tensors=14 baseline_source_shared=True saved_ptrs_unique=True
```

A corrected read-only load check on the completed artifact also passed: 14
keys, `primary_activations=(5,8,2048)`,
`repeated_endpoint_activations=(4,3,2048)` for the source-declared
`REPEATED_CONTEXT_INDICES=(0,1,5,6)`, baseline equality, and diagnostics
artifact hash consistency.

## Run4 completed attempt

Run4 used the same predeclared panel and command with the serialization fix:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /usr/bin/time -v -o /tmp/trr-p02-time-20260905-run4.txt timeout --foreground 120s python3 scripts/trr_p02/diagnose_geometry.py --plan experiments/TRR-P02/plan.json --prototype /tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors --historical-lens /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --output-root experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4 --manifest-path experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4/run_manifest.json --implementation-commit 470b6f1becfaa6da110048302938feddd7204c30 --prototype-chunk-size 8192
```

The process started at `2026-09-05T12:23:25Z`, ended at
`2026-09-05T12:23:37Z`, exited `0`, did not time out, used eight Torch
intra-op threads and one inter-op thread with CUDA hidden, and had outer Max RSS
`6,324,564 kB` with zero swaps. The internal diagnostics peak was
`6,476,353,536` bytes and all five 8-GiB RSS checks passed. The 10-GiB
available-memory preflight guard passed.

Run4 retained `preflight.json`, `qualification.json`,
`activation_panel.safetensors`, both figures, `diagnostics.json`,
`run_manifest.json`, and copied outer stdout/stderr/time/receipt files under
its output root. Key hashes are recorded in `run_manifest.json`; the primary
artifacts are: activation panel
`e63026f56063083fe009fe3211548875310dd3295e7c205f0e3759f1ae5a15ca`,
diagnostics `7352573df457804b2702a419571a9feb100ae5863d32238eab6f38f19a9586c4`,
manifest `2ad5a6049c988940acbe0e1ef4b62320ad094c1b2e1673ca6f8e5edcc7f7f710`,
offset figure `6796c56602983dac2340e6ae8992e7adc2246ff621908bbb0c3b46d07129e721`,
and lens figure `fc03d6f48d19c694ac7bc5bbac716b08ad85e9bc95881a61a2265cb217698cc7`.

The initial run4 shell wrapper failure is preserved as
`experiments/TRR-P02/runtime/cpu-public-geometry-20260905-run4-launch-failure.json`
(SHA256 `6295147889eb6a8aa980d59af7255f9cf2eca7b8f9a81ce9f79490447370c273`);
it exited before Python started because redirection targeted the runner's
create-only output directory. It opened no model or table and was followed by
the successful run4 invocation above.

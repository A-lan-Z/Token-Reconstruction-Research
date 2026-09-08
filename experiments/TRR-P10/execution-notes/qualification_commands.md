# Read-only qualification commands (not executed)

These commands are a concrete qualification plan for already-opened PR20 and TRR-0009
A1+A2 fixtures. They do not select sources, load labels, or score truth. Every
output path is create-only and the trace output is directed to `/tmp`, outside
the source worktree.

The existing wrapper equivalence receipt is
`/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010/experiments/TRR-0010/evaluation/a1_a2_opened_fixture_equivalence_v3/result.json`.
It passed direct native `propose_public_a1 -> decode_policy` versus the
TRR-0010 native A1+A2 wrapper for one public-base row in each of Pile and
Finance. Its diagnostic source SHA-256 is
`1abe9a51ba718690887a378e03c3d16c08bcbdad94e1436c5988956605dc03e5`; the
legacy adapter SHA-256 is
`36f6aa7b4493c60b257b3896c975f523595da912e14a97dba3f6440d419e8427` and the
TRR-0010 runner SHA-256 is
`90f3fe55ef9f714decfa12110d88b8d4f33c39e968e3a2153c566cebe01ef696`.

The narrow P10 qualification wrapper is `scripts/trr_p10/qualification.py`. It
imports the hash-bound PR20 registration/runner and wraps
`trr0003_footing_compare.propose_public_a1` at the actual A1 decision point. It
loads the first one or two rows from each already-opened public-base
observation cell, runs trace-off and trace-on in separate fresh adapters, and
requires exact trace-on/off IDs plus exact equality to the frozen
`frozen_a1_a2_k256` predictions. It writes proposal IDs/scores
`[records*cells,128,512]` to a create-only safetensors trace under `/tmp`. The
fixed decoder top-256 trace remains explicitly unavailable.

Run only after root leases the GPU, from any worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /tmp/trr-p10/scripts/trr_p10/qualification.py \
  --repository-root /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010 \
  --registration /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010/experiments/TRR-0010/evaluation/registration_r5.json \
  --freeze /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010/experiments/TRR-0010/evaluation/public_prediction_watchdog_r5/public_freeze.json \
  --output-root /tmp/trr-p10-runtime/a1-trace-qualification-r1 \
  --cells pile__public_base finance__public_base \
  --records 1 \
  --device cuda \
  --max-seconds 180
```

The prior historical diagnostic remains a read-only reference; its output root
must not be reused. The trace artifact should be hash-bound, truth-free, and
create-only under `/tmp/trr-p10-runtime`.

The immutable Agent1 package can qualify the fixed replay path independently.
Run from the TRR-0011 worktree, using one process per method/cell and a new
receipt path:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/trr0011_package.py replay \
  --manifest experiments/TRR-0011/package/inference_package_v2.json \
  --root . \
  --method expanded_fixed \
  --cell pile__public_base \
  --max-records 2 \
  --device cuda \
  --output /tmp/trr-p10-runtime/pkg-expanded-pile-r1.json
```

Repeat only for `current_fixed` and one Finance cell after the first receipt
passes. Package SHA-256 is
`a74f687f8287cacfeafa0acc81a97f208672dedf797792c60f88ba93b0318b25`.

Both paths must fail closed before loading resources if the live guard is not
met: free GPU memory at least 8 GiB, CUDA reserved memory at most 6 GiB for the
A1+A2 qualification, host RSS at most 16 GiB, host available memory at least
10 GiB, disk free space at least 20 GiB, and wall time at most 180 seconds.
The A1+A2 historical peak basis is 3.303 GiB; retain a margin for the display
allocation and report measured peak/reservation. This note records commands
only; no qualification run was launched in P10 preparation.

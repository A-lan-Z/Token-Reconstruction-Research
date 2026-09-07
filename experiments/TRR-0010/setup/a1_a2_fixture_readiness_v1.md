# TRR-0010 A1+A2 opened-fixture equivalence readiness

Status: prepared, not launched. This note is metadata-only. No H tensor,
model, H tensor, candidate array, target label, source text, or truth payload
was materialized by this preparation step; the published prediction fixture was
only identity/header/hash inspected.

## Fixture identity

The published A1+A2 prediction fixture is the four
experiments/TRR-0003/evidence/predictions/<domain>/<condition>/frozen_a1_a2_k256.safetensors
artifacts bound to panel
experiments/TRR-0003/footing/panel.json (panel SHA-256
d1810330f53ebb7e149be45b8c07414e09d30b31b68c7c09d24df3854fec7333). Its
panel has eight Pile and eight Finance records. Aggregate comparison of its
public-record hashes with the frozen TRR-0009 selection-v2 metadata found zero
identity overlap in both domains (TRR-0003: 8/domain; TRR-0009: 128 Pile and
256 Finance). Therefore those prediction files cannot be used as an exact
same-record oracle for the opened TRR-0009 H rows.

The already-opened TRR-0009 public observation manifest is
experiments/TRR-0009/evaluation/public_observations_v2/observations.json
(status FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH). Its public-base observations
are:

- Pile: 128 x 128 x 2048 BF16 H, SHA-256
  5ad0fece58e4247d8fb1a1d8f5d1d5a021eb96cded0be112f0aafa9052dab89a.
- Finance: 256 x 128 x 2048 BF16 H, SHA-256
  7449bf11fd335ec8d46ca7581378b8e3a3351d7b7852561cac944c723dd643bb.

The planned diagnostic is a direct native-versus-runner adapter comparison on
only the first public-base H row per domain. It checks exact token-ID equality
after the runner's normalization and records only aggregate digests.
It is a loader/path equivalence qualifier, not an accuracy comparison.

## Frozen native bindings

The reusable production callable is
scripts/trr0010_eval_runner._load_native_a1_a2. It imports the retained
native adapter from scripts/trr0004_predict_confirmation and the fixed policy
from scripts/trr0003_footing_compare._fixed_k256_policy. The pinned operation
is one composite method frozen_a1_a2_k256:

- public Alpaca lens SHA-256
  33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742;
- normalized public E SHA-256
  ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1;
- public Llama snapshot revision 9213176726f574b556790deb65791e0c5aa438b6,
  model SHA-256
  1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f;
- public reference helper SHA-256
  10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd;
- proposal budget 512, proposal chunk 256, fixed candidate K 256,
  direct-cosine policy a1a2_43ea0bb737bc075531ca, reconstructed prefix,
  candidate arrays memory-only.

The task-local asset binding receipt is
experiments/TRR-0010/setup/a1_a2_public_runtime_descriptor_v1.json.
It binds the lens, public E, prefix snapshot files, reference helper, and
native source hashes. The current TRR-0010 runner source is SHA-256
90f3fe55ef9f714decfa12110d88b8d4f33c39e968e3a2153c566cebe01ef696.

## Guarded command

The bounded opened-fixture diagnostic is now implemented at
scripts/trr0010_a1_a2_opened_fixture_equivalence.py. It compares one
public-base H row per domain through the direct retained native proposal/decode
helpers and the existing TRR-0010 _load_native_a1_a2 wrapper. Both paths share
one loaded public model; the receipt therefore reports wrapper input/causal-state
and output equivalence, not independent algorithm accuracy.

The exact command, pending root's exclusive GPU lease, is:

    cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:src:scripts timeout --signal=TERM --kill-after=15s 180s python3 scripts/trr0010_a1_a2_opened_fixture_equivalence.py --repository-root . --descriptor experiments/TRR-0010/setup/a1_a2_public_runtime_descriptor_v1.json --observation-manifest /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0009/experiments/TRR-0009/evaluation/public_observations_v2/observations.json --output-root experiments/TRR-0010/evaluation/a1_a2_opened_fixture_equivalence_v2 --device cuda --max-seconds 180

The diagnostic itself guards 8 GiB minimum free GPU memory before load, 6 GiB
maximum CUDA reserved memory, 16 GiB maximum host RSS, GPU exclusivity and
temperature below 85 C, host MemAvailable, disk space, and the 180-second
wall bound.

For the eventual full matrix, after root freezes the TRR-0010 panel and
registration, the existing runner's source-bound command is:

    cd /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:src:scripts timeout --signal=TERM --kill-after=15s 1800s python3 scripts/trr0010_eval_runner.py --repository-root . --registration experiments/TRR-0010/evaluation/registration.json --device cuda

This is the existing truth-free runner and includes the bound native A1+A2
method in its six-method matrix. Its A1+A2 loader applies the reconstructed
prefix and fixed 512-to-256 policy. The runner's registered observation
bindings must point to the final TRR-0010 panel; the current TRR-0009
public-base observations are available for the one-row-per-domain qualifier but
are not a final TRR-0010 registration.

Use the established TRR-0004 guard limits for the bounded qualifier/run:
8 GiB minimum free GPU memory before load, 6 GiB maximum CUDA reserved memory,
16 GiB maximum host RSS, GPU exclusivity and temperature below 85 C, and a
180-second wall limit for the one-row-per-domain direct adapter check. The
published native 128-record timing fixture reached about 3.55 GiB reserved
and 5.82 GiB RSS, so these limits retain margin; the full 256-Finance/128-Pile
matrix requires a separate prospective budget and release.

No GPU command was launched. This opened-fixture diagnostic binds the already
opened TRR-0009 public observation manifest directly and does not require a
final TRR-0010 registration. The separate full matrix command above still
requires its final panel and A1+A2 resource registration; no historical
prediction file is substituted for the missing same-record fixture.

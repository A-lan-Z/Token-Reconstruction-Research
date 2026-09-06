# TRR-P03 Stage 1 run plan

**Scope:** this is the implementation-side execution plan for the frozen
Stage-1 natural diagnostic. It contains no truth-derived result and does not
open the Stage-2 holdout. Root substitutes the post-review implementation
commit in the commands after committing the source.

## Fixed inputs and order

Use `/tmp/trr-p03` as the working directory and run every numeric command under
`scripts/trr_p03/resource_watchdog.py`. The watchdog thresholds are exactly
`max-rss-bytes=8589934592` (8 GiB group RSS),
`min-available-bytes=10737418240` (10 GiB host `MemAvailable`), and a
create-only receipt directory. The child environment is:

```text
CUDA_VISIBLE_DEVICES=
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
TOKENIZERS_PARALLELISM=false
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONPATH=.:src:scripts/trr_p01
```

The read-only public assets are:

```text
prototype = /tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors
prototype_sha256 = 51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3
lens = /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt
lens_sha256 = 33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742
```

The exact lens hash above is pinned by the setup and source receipts as
`33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`.
The frozen public matched model is
`/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6`.
The evaluator-only shifted target is
`/home/alanz/.cache/huggingface/hub/models--Vikhrmodels--Vikhr-Llama-3.2-1B-Instruct/snapshots/7fa9d06a59246629244cdd3b6b92e4fc756baa0f`.

The shared projected preparation already passed as
`experiments/TRR-P03/runtime/projected-preparation-r1`, with projected table
SHA-256
`8fa4e65ca5ae0c4492c16290403f38126894f5d41383bd2e2b178fbb85003ba7` and
file size `1050673728` bytes. Every reconstruction command below passes that
same file:

```text
/tmp/trr-p03/experiments/TRR-P03/runtime/projected-preparation-r1/projected_prototypes.safetensors
```

The exact A1+A2 anchor declaration is
`experiments/TRR-P03/review/anchor-record-ids.json`: length-39 records
`p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`, and `p03-s1-r0012`, selected at
within-stratum indices `[0,2,4,5]` before observations or truth.

## Geometry and estimates

Stage 1 has 24 records per target, six at each post-BOS length 16, 39, 64,
and 128. The sequence slots including BOS are 17, 40, 65, and 129, so each
arm has `6*(16+39+64+128)=1482` scored positions and the paired matrix has
2964 positions. The model uses `H=2048`, `V=128256`, cut depth 4, BOS 128000,
BF16 observations, and float32 scoring.

With query chunks of 256 and prototype chunks of 8192, each full-vocabulary
arm has six query chunks and 16 candidate chunks: 96 score blocks per raw
method and 96 per projected method. Both arms therefore have 192 raw and 192
projected blocks. Standalone native A1 has six full-vocabulary lens forwards
per arm (12 paired); all decisions retain the positive `exp(s)`-scaled native
lens logits and report the scale separately. The fixed A1+A2 anchor uses
exactly four records and exact 40-slot geometry: `4*39*256=39936` candidate
simulations per arm, one A1 proposal forward over 156 post-BOS queries, and
79 public-prefix calls with record batch size 8 (`1 + 2*39`). Its persisted
candidate and score arrays are each only `4*40*256*4=163840` bytes.

The raw boundary table is 525,337,024 bytes on disk (BF16 tensor payload
525,336,576 bytes). A projected full table is
`128256*2048*4=1050673152` float32 tensor bytes and the prepared safetensors
file is 1,050,673,728 bytes. A normalized float32 candidate table has the same
1,050,673,152-byte payload; the implementation constructs it once per readout
call and reuses it across query chunks. The maximum score scratch at query
256 is `256*128256*4=131334144` bytes. The decoder weight file is about
2.47 GB; the prior identical P02 full run peaked at 5,239,828 KiB and the
shared projected preparation measured 4,026,428 KiB internally and
3,335,843,840 bytes in the watchdog group. These observations leave a
substantial margin below the 8 GiB process ceiling; each new run still has to
pass the live watchdog and preserve its sampled minimum host availability.

A clean qualification requires child and wrapper exit 0, readable watchdog
samples, peak group RSS below 8 GiB, minimum sampled host availability at
least 10 GiB, and no CUDA allocation. If a resource workaround changes the
chunk partition, its qualifier must also pass the numeric output-equivalence
check described below. A failed guard, timeout, allocator failure, or
non-equivalent batching attempt is retained as excluded development evidence
and cannot trigger a score rerun or method change.

## Commands

`P03_COMMIT` below is the exact source commit written into all receipts after
root freezes this branch. The preparation command has already run with source
commit `23933e65868dcdfa58a44ce47b87a8f5a7455c51`; do not regenerate it under a
different output path.

First, generate a create-only truth-free qualification subset for each target.
The evaluator command may read its panel/private sidecar, but only its
`public/observation_index.json` is passed to reconstruction. The subset is
predeclared in `qualifier-record-ids.json` and contains all six length-128
records plus the four anchors. Never use `--stage all`.

```text
# bundle-a: matched public target
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/qualifier-generation-a \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label qualifier-generation-a -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/generate_observations.py \
  --panel experiments/TRR-P03/setup/panel-20260906-frozen/stage1/evaluator_panel.json \
  --record-ids experiments/TRR-P03/review/qualifier-record-ids.json \
  --output-root experiments/TRR-P03/runtime/qualifier-observations-bundle-a \
  --bundle-id bundle-a --stage stage1 \
  --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --device cpu --batch-size 4 --seed 20260906 \
  --required-bytes 10737418240 --expected-peak-bytes 8589934592 \
  --implementation-commit P03_COMMIT

# bundle-b: shifted full-SFT target; condition mapping remains evaluator-only
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/qualifier-generation-b \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label qualifier-generation-b -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/generate_observations.py \
  --panel experiments/TRR-P03/setup/panel-20260906-frozen/stage1/evaluator_panel.json \
  --record-ids experiments/TRR-P03/review/qualifier-record-ids.json \
  --output-root experiments/TRR-P03/runtime/qualifier-observations-bundle-b \
  --bundle-id bundle-b --stage stage1 \
  --model-path /home/alanz/.cache/huggingface/hub/models--Vikhrmodels--Vikhr-Llama-3.2-1B-Instruct/snapshots/7fa9d06a59246629244cdd3b6b92e4fc756baa0f \
  --device cpu --batch-size 4 --seed 20260906 \
  --required-bytes 10737418240 --expected-peak-bytes 8589934592 \
  --implementation-commit P03_COMMIT
```

The qualifier reconstruction then runs all three full-vocabulary methods and
the native four-record A1+A2 anchor for both target bundles. The readout model
for A1/A2 is always the pinned public matched checkpoint, even for bundle-b;
the shifted checkpoint was used only to create bundle-b observations.

```text
METHODS=raw_boundary.cosine,projected_boundary.cosine,historical_a1.cosine,historical_a1_a2_anchor.cosine

# bundle-a qualifier
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/qualifier-reconstruction-a \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label qualifier-reconstruction-a -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/reconstruct.py \
  --observation-index experiments/TRR-P03/runtime/qualifier-observations-bundle-a/public/observation_index.json \
  --prototype /tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors \
  --historical-lens /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt \
  --projected-prototype experiments/TRR-P03/runtime/projected-preparation-r1/projected_prototypes.safetensors \
  --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --anchor-records experiments/TRR-P03/review/anchor-record-ids.json \
  --methods "$METHODS" --output-root experiments/TRR-P03/runtime/qualifier-reconstruction-bundle-a \
  --device cpu --query-chunk-size 256 --prototype-chunk-size 8192 \
  --plan experiments/TRR-P03/plan.json --seed 20260906 --implementation-commit P03_COMMIT

# bundle-b qualifier: only the observation index and output root differ; the
# reconstruction model remains the pinned public matched base
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/qualifier-reconstruction-b \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label qualifier-reconstruction-b -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/reconstruct.py \
  --observation-index experiments/TRR-P03/runtime/qualifier-observations-bundle-b/public/observation_index.json \
  --prototype /tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors \
  --historical-lens /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt \
  --projected-prototype experiments/TRR-P03/runtime/projected-preparation-r1/projected_prototypes.safetensors \
  --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --anchor-records experiments/TRR-P03/review/anchor-record-ids.json \
  --methods "$METHODS" --output-root experiments/TRR-P03/runtime/qualifier-reconstruction-bundle-b \
  --device cpu --query-chunk-size 256 --prototype-chunk-size 8192 \
  --plan experiments/TRR-P03/plan.json --seed 20260906 --implementation-commit P03_COMMIT
```

The full Stage-1 matrix uses fresh wrapper and child roots. These are the
concrete generation commands after the qualifier passes:

```text
# bundle-a full Stage-1 observations
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/stage1-generation-a \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label stage1-generation-a -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/generate_observations.py \
  --panel experiments/TRR-P03/setup/panel-20260906-frozen/stage1/evaluator_panel.json \
  --output-root experiments/TRR-P03/runtime/stage1-observations-bundle-a \
  --bundle-id bundle-a --stage stage1 \
  --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --device cpu --batch-size 4 --seed 20260906 \
  --required-bytes 10737418240 --expected-peak-bytes 8589934592 \
  --implementation-commit P03_COMMIT

# bundle-b full Stage-1 observations
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/stage1-generation-b \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label stage1-generation-b -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/generate_observations.py \
  --panel experiments/TRR-P03/setup/panel-20260906-frozen/stage1/evaluator_panel.json \
  --output-root experiments/TRR-P03/runtime/stage1-observations-bundle-b \
  --bundle-id bundle-b --stage stage1 \
  --model-path /home/alanz/.cache/huggingface/hub/models--Vikhrmodels--Vikhr-Llama-3.2-1B-Instruct/snapshots/7fa9d06a59246629244cdd3b6b92e4fc756baa0f \
  --device cpu --batch-size 4 --seed 20260906 \
  --required-bytes 10737418240 --expected-peak-bytes 8589934592 \
  --implementation-commit P03_COMMIT
```

The full reconstruction commands use separate watchdog roots and the pinned
public base checkpoint for both arms:

```text
# bundle-a full reconstruction
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/stage1-reconstruction-a \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label stage1-reconstruction-a -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/reconstruct.py \
  --observation-index experiments/TRR-P03/runtime/stage1-observations-bundle-a/public/observation_index.json \
  --prototype /tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors \
  --historical-lens /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt \
  --projected-prototype experiments/TRR-P03/runtime/projected-preparation-r1/projected_prototypes.safetensors \
  --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --anchor-records experiments/TRR-P03/review/anchor-record-ids.json \
  --methods "$METHODS" --output-root experiments/TRR-P03/runtime/stage1-reconstruction-bundle-a \
  --device cpu --query-chunk-size 256 --prototype-chunk-size 8192 \
  --plan experiments/TRR-P03/plan.json --seed 20260906 --implementation-commit P03_COMMIT

# bundle-b full reconstruction
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/stage1-reconstruction-b \
  --cwd /tmp/trr-p03 --timeout-seconds 900 --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 --min-available-bytes 10737418240 \
  --label stage1-reconstruction-b -- \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:src:scripts/trr_p01 \
  python3 scripts/trr_p03/reconstruct.py \
  --observation-index experiments/TRR-P03/runtime/stage1-observations-bundle-b/public/observation_index.json \
  --prototype /tmp/trr-p01/experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors \
  --historical-lens /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt \
  --projected-prototype experiments/TRR-P03/runtime/projected-preparation-r1/projected_prototypes.safetensors \
  --model-path /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --anchor-records experiments/TRR-P03/review/anchor-record-ids.json \
  --methods "$METHODS" --output-root experiments/TRR-P03/runtime/stage1-reconstruction-bundle-b \
  --device cpu --query-chunk-size 256 --prototype-chunk-size 8192 \
  --plan experiments/TRR-P03/plan.json --seed 20260906 --implementation-commit P03_COMMIT
```

The canonical q256/p8192 partition is the preregistered run and is not being
used as a resource workaround, so no alternative partition is required. If a
resource workaround later changes query or candidate batching, repeat that
qualifier into a distinct create-only root and compare the numeric prediction,
diagnostic, candidate-set, and JSONL row values against the canonical output.
Compare tensor values directly when hashes differ; timing/preflight/evidence
receipts are expected to differ. Any numeric mismatch excludes the changed
partition and stops the matrix.

Only after both qualifier arms pass the watchdog may the explicit full-matrix
commands above be run. If a resource workaround changed batching, its numeric
output-equivalence check must also pass before proceeding. These two prediction
roots must be immutable and complete before truth is opened.

Before the first read of
`experiments/TRR-P03/setup/panel-20260906-frozen/stage1/private_truth.jsonl`,
run the separate strict design-owned validator over exactly those two full
prediction roots. It must write an immutable PASS receipt covering the exact
ordered 24 IDs and lengths, all three static methods over all 24 records, the
four-anchor A1+A2 method, both observation indexes and their mask/position
digests, matching plan/source/config/public assets, and the bundle-a/base and
bundle-b/Vikhr evaluator mapping. The final scorer invocation must supply both
prediction roots and this PASS receipt and must not use an unequal-strata
escape. The strict validator’s final CLI is recorded by its owner before root
starts the scoring command.

After that PASS receipt exists, open the Stage-1 JSONL truth once and run the
separate scorer with bootstrap draws `10000`, seed `20260905`,
`records-per-stratum=6`. It writes numeric `per_record.jsonl`,
`paired_statistics.json`, and `metrics.json` before any optional plotting. The
score output must include both target arms, all static methods, anchor coverage,
length and position summaries, native A1 score units and `exp(s)` scale, and
paired projected-vs-A1 token gains (`projected correct/A1 wrong`), regressions,
both-correct, and both-wrong counts from saved correctness vectors. Style labels
remain evaluator-side panel metadata and are joined only by the root if a
style-stratified summary is needed. Stage-2 truth remains sealed regardless of
Stage-1 outcome.

The generic scorer’s model-free smoke already exercises save/freeze/score
ordering, so a failure in any plotting or presentation step cannot erase the
numeric score files. No plot is an input to the freeze or gate.

The concrete strict joint-gate command, after the full prediction roots are
complete, is:

```text
python3 scripts/trr_p03/validate_stage1.py \
  --plan experiments/TRR-P03/plan.json \
  --observation-index-a experiments/TRR-P03/runtime/stage1-observations-bundle-a/public/observation_index.json \
  --observation-index-b experiments/TRR-P03/runtime/stage1-observations-bundle-b/public/observation_index.json \
  --prediction-root-a experiments/TRR-P03/runtime/stage1-reconstruction-bundle-a \
  --prediction-root-b experiments/TRR-P03/runtime/stage1-reconstruction-bundle-b \
  --resource-mapping experiments/TRR-P03/setup/panel-20260906-frozen/plan-input.json \
  --resource-receipt-a experiments/TRR-P03/runtime/stage1-observations-bundle-a/generation_evidence.json \
  --resource-receipt-b experiments/TRR-P03/runtime/stage1-observations-bundle-b/generation_evidence.json \
  --watchdog-receipt-a /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/stage1-generation-a/finish.json \
  --watchdog-receipt-b /tmp/trr-p03/experiments/TRR-P03/runtime/watchdog/stage1-generation-b/finish.json \
  --implementation-commit P03_COMMIT \
  --output-root experiments/TRR-P03/runtime/stage1-joint-validation
```

The command has no truth argument. Its create-only output must be the sole
`joint_validation_receipt.json` supplied as `--pre-score-receipt` to the
scorer; use the Stage-1 JSONL sidecar only in that subsequent scorer command.

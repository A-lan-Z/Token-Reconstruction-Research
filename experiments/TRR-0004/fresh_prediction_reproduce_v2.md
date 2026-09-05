# TRR-0004 V2 prediction reproduction

This note is the task-local companion to `fresh_confirmation_reproduce.md`.
The latter records the public selection and capture chain; this note records
the completed, five-process V2 prediction run and the restore boundary. It is
an exploratory confirmation record, not a replacement claim.

## Frozen run

The prediction source was frozen at commit
`a11b37bbe11b429b07ba82074906c1698295bd3f`. The registered predictor is
`scripts/trr0004_predict_confirmation.py`, 71,405 bytes, SHA256
`36f6aa7b4493c60b257b3896c975f523595da912e14a97dba3f6440d419e8427`.
The V2 method registration is
`fresh_confirmation_v1/panel_capture/registration_v2.json`, SHA256
`543af6d1c57854996c420d2a342e9121cf01b8b444599aa93c514ba18bd968bb`.
It binds the panel SHA256
`da65242f395c2c96a25ed8e30d62415db9108c9ded0c9525d4f9358691cb44da` and
the selection-plan SHA256
`a01f445c67bb4cb0f462fb000dd8a46c63cceeebe4c53c1e73c08107378f3eca`.

The exact argv ledger is
`fresh_prediction_v2_argv.json`, SHA256
`54b73566c6789d3d19317680ca4f70f8d33ab9a44d721b097bd8f2307975af28`.
The run used the byte-identical repository copy of the mechanical launcher,
`evidence/prediction_v2_mechanical_launcher.py`, SHA256
`f44d964bf64019c78026953bc76c22fb50d220d63334171612c9d5c501490d3d`.
The original launcher was in `/tmp`; the receipt records both paths and
hashes. Its completion receipt is
`evidence/prediction_v2_mechanical_launcher_receipt.json` and the compact
per-method summary is
`evidence/prediction_v2_receipt_summary.json`.

## Exact mechanical invocation

Run this from a clean restore of the worktree at the frozen source commit.
The five method output roots are create-only. Do not rerun in the existing
worktree while those roots exist; use a clean worktree at the same path or
derive a new argv ledger with only fresh output-root paths changed. The
recorded ledger and receipt remain the authoritative record of the completed
run.

```bash
ROOT=/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004
PY=$ROOT/.venv-trr0004/bin/python
cd "$ROOT"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONPATH=src:scripts "$PY" \
  experiments/TRR-0004/evidence/prediction_v2_mechanical_launcher.py \
  experiments/TRR-0004/fresh_prediction_v2_argv.json
```

The launcher mechanically reads the five argv lists and invokes them in this
order: `frozen_a1_a2_k256`, `historical_alpaca_a1`,
`historical_affine_ce_no_vocab_bias`, `causal_h_attention128`, and
`positionwise_mlp256`. Each process uses the same four public cells, one
warmup and three measured calls per record, validates four complete outputs,
and writes only to its own fresh root. Standalone methods have no A2 fallback;
the retained A1 method emits top-1 directly and has no candidate arrays. A2
retains its native candidate outputs and public-prefix simulations.

This invocation opens no evaluator-private truth. After all five processes
finish, the footing freeze path must validate completeness, integrity,
panel/method/config/code bindings, and exact prediction hashes before its
separate score path opens the truth sidecar. The canonical aggregate is
`fresh_confirmation_v1/predictions_v2`; the ignored per-method roots are
execution scratch/output roots, not the canonical archive.

## Inputs and restore boundary

Restore the metadata and compact binaries listed in
`evidence/staging_readiness_v1.json`. The four panel observation tensors are
the exact public inputs for the 20 predictions. Restore the retained A1 lens,
the four affine selected states, the two contextual selected states, and the
contextual position schedule from the evidence paths listed there. The
aggregate prediction tensors and their run-evidence JSON are already copied
under `fresh_confirmation_v1/predictions_v2`.

The following public resources remain external and are validated by the
registered hashes:

* the Llama public checkpoint blob
  `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`;
* the public prefix config blob SHA256
  `2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb`;
* the normalized embedding table SHA256
  `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`;
* the public LoRA update used only during observation generation, SHA256
  `eea7bb49f801b61df2e26a8f59af7c3096f6f3a2604404e16e589443bcfba595`;
* the public Track B fit activation artifact, SHA256
  `d1c78fcf1acc91b57d51355ee11f267bf4c12f1bc7d5160164b3b6ea11b45344`,
  and its registered mixed validation artifact, SHA256
  `a8e7633ffb369864af33754c5ebb2d9a4ca9d6e7d4550731e8ff26e20c8200cf`.

The model, embedding, LoRA, and fit activation payloads are deliberately
hash-only because they are large or preparation-only. The model checkpoint
and config are bound in every method registration for a common startup
integrity gate, but only A2 loads and uses public-prefix model math during
prediction. The truth sidecar is also deliberately external and is permitted
only after the complete prediction gate.

To regenerate missing public preparation artifacts, use the commands in
`fresh_confirmation_reproduce.md` with the pinned local caches and the
recorded source/resource hashes. To replay Track B fitting, restore the
external fit activations plus `fit/adapter_v2/public_fit_manifest.json` and
the registered fit receipts; the selected states in evidence are sufficient
for the scored inference replay and do not require refitting.

## Exact restore procedure

Keep the final handoff checkout as a read-only archive. Create a separate
execution worktree at source commit
`a11b37bbe11b429b07ba82074906c1698295bd3f`; the final handoff commit
necessarily comes later because it contains the evidence archive. Copy the
staged panel metadata/observations, method configs, and comparator files from
the archive checkout into that execution worktree. The registered affine and
contextual state paths point to their original `outputs/TRR-0004/...`
locations, so reverse-copy the selected `.safetensors` entries in
`experiments/TRR-0004/evidence/affine/affine_copy_map_v1.json` and
`experiments/TRR-0004/evidence/contextual/copy_hashmap.json` from each
archive `destination` to its recorded `source`, creating parent
directories and verifying the recorded SHA256; do not overwrite a differing
file. The retained A1 lens is already at its registered
`experiments/TRR-0004/evidence/comparators/public_a1_lens.pt` path. The
recorded argv contains absolute paths to the original worktree: use that exact
ledger only when the clean execution worktree has that path; otherwise derive
a relocated ledger changing only root-dependent paths and keep all method
arguments, hashes, and registration semantics unchanged.

Public preparation is reproducible from the exact command records at
`evidence/public_activation_launch_v2/launch.json` and
`evidence/producer_v1/{selection,capture,register_v2}/`; the public fit
commands are in `fit/fit_small_v1.receipt.json`,
`fit/fit_large_v1.receipt.json`, and
`evidence/contextual/argv/contextual_fit_v1_argv.json`. These receipts and
the pinned resource hashes are the authoritative regeneration commands; the
truth preparation/score records remain a separate post-gate role. The successful public activation receipt is also byte-copied at `experiments/TRR-0004/evidence/archived_preparation_v1/public_activation_v2/preparation_evidence.json`; its hash is recorded in `archived_failures_v1/copy_map.json` while the large activation tensor remains external.

## Excluded attempts and known gaps

`evidence/archived_failures_v1/copy_map.json` preserves byte-identical small
receipts for the V1 A2 embedding-mismatch failure, the V1 public-activation
padding failure, contextual qualification failure, and excluded affine
qualification. `evidence/archived_successes_v1` preserves the successful V1
standalone A1/no-vocabulary-bias outputs as excluded integration evidence.
Original ignored output roots remain in place and were not overwritten or
deleted. The excluded qualification partial states are recorded by hash in
`EXCLUDED_QUALIFICATION.json` but are not needed by, or eligible for, the
final selection.

The original historical A1 fit row IDs and measured preparation cost are not
available; the retained state is therefore a faithful implementation bridge
with a provenance limitation. Reproduction of the exact selected state uses
the staged state bytes. Reproduction of the fit itself requires the external
public activation artifacts and the registered split/fit metadata. Empty log
files are retained in the force-add list where receipts reference them;
nonempty logs and all compact failure metadata are likewise listed in the
staging map.

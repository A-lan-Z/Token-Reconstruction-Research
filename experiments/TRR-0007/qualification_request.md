# TRR-0007 enriched-bank qualifier request

Status: prepared, not launched. Root must grant the exclusive GPU lease before
running either command.

The bounded qualifier uses the actual retained enriched bank and the trainer's
full representative gathered batch. It runs both registered methods for two
updates, with the common neutral initialization, shared seed-4005 schedule,
full-vocabulary public normalized E, validation selection, and the initially
wrong challenge subset.

    env PYTHONPATH=.:src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python3 scripts/trr0007_train_positionwise.py --banks enriched --qualification-only --qualification-steps 2 --device cuda --fit-manifest experiments/TRR-0005/public_activation_v1/enriched_manifest.json --validation-manifest experiments/TRR-0005/public_activation_v1/enriched_manifest.json --retained-reference experiments/TRR-0005/joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors --output experiments/TRR-0007/qualification_enriched_v1

Metadata-only preflight observed:

- fit H storage [1200,192,2048], validation H [48,192,2048], public E
  [128256,2048];
- every gathered update materializes H [8,192,2048], masks/labels [8,192],
  and selected full-vocabulary logits [512,128256];
- current diagonal model has 5,247,361 parameters; the residual adds
  2,099,712 and totals 7,347,073;
- analytical peak is 1,726,199,824 bytes; the enforced conservative floor is
  4,413,456,384 bytes (1.5 times the measured TRR-0005 2,942,304,256-byte
  peak), with a 10 GiB host-available guard and 8 GiB free-GPU guard.

This two-step qualifier uses a two-step cosine schedule because its purpose
is resource and geometry qualification; its learning curve is not treated as
a prefix-equivalent main fit.  After the qualifier completes and its
peak-memory/geometry receipts pass, the old-bank full fit uses the same
command with --qualification-only --qualification-steps 2 replaced by
--steps 3000, and a new output root.  The later improved-bank fit uses
--banks improved with the same 3,000-step schedule, neutral initialization,
optimizer, schedule seed, and geometry; --banks both is reserved for a clean
complete crossed rerun.

# TRR-0004 setup and preflight

Status: **public activation preparation complete; no confirmation records generated**.

Recorded at 2026-09-05T10:09:56.574088+00:00. The isolated worktree is `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004` on `task/TRR-0004`, exactly at `eab3fc21fdae67fe628a42620029e25829a188b1`. The incoming packet is preserved byte-for-byte at `coordination/requests/TRR-0004.md` (8,551 bytes; SHA256 `7eb85bc38225b253fe4a0410961539130882277d64dff5e6ba295766a4b197d0`).

## Runtime

- Python: `/usr/bin/python3`, 3.12.3; Torch `2.10.0+cu128`; Transformers `5.3.0`; datasets `4.8.3`; safetensors `0.7.0`; pytest `8.4.2`.
- Host: AMD Ryzen 9 9950X3D 16-Core Processor, 32 logical threads, 30.2 GiB total RAM and 21.6 GiB available at the probe.
- GPU probe: `NVIDIA GeForce RTX 5080, 16303 MiB, 3993 MiB, 11985 MiB, 50`. Torch reported `True` with `1` device(s). This is an observation, not an allocation; root controls the next GPU slot.
- Disk at the worktree filesystem: 295.3 GiB free of 1006.9 GiB.
- Network remains offline/local-cache only, and no global package installation or paid compute was used.

## Reusable assets and roles

The pinned Llama 3.2 1B Instruct snapshot is available at `/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6` (2,482,971,357 bytes across eight members), with member digests recorded in the prior manifest. The pinned `NeelNanda/pile-10k` cache blob is present at `/home/alanz/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/blobs/a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31` (33,262,901 bytes; content digest matches the registry). Existing cut-4 public observations, the synthetic public LoRA-shift observations, the retained A1 lens, the direct inverse, and TRR-0003 decoder states are available through the shared root outputs and remain read-only; none was copied into this worktree.

TRR-0003’s panel metadata establishes cut depth 4, hidden size 2048, Pile sequences of 40 positions, and Finance sequences of 128 positions. Those records are retrospective development material and are not reserved as new held-out TRR-0004 evaluation data. The evaluator-private TRR-0003 truth sidecar was not accessed or copied. TRR-P01’s separate worktree and outputs were not touched.

## Proposed bounded resource envelope

Before any run, root should grant one GPU process and require a largest-geometry qualification first. The preliminary fail-closed envelope is at least 8 GiB observed free GPU memory before launch, at most 8 GiB reserved GPU memory (leaving 8 GiB on the 16 GiB device), at most 16 GiB host RSS, eight CPU workers, and at most 10 GiB task output. Any OOM, non-finite result, driver/allocator anomaly, or provenance mismatch stops the process and preserves the failed attempt. The primary path is fixed batch 8 × 192. Its qualification changes only future pad token IDs in the same batch shape and requires bit-exact active outputs; the batch-1 unpadded comparison is a diagnostic and cannot authorize a substitute path. The preparation command is bounded by a 600-second timeout.

A later experiment should first perform the cheap A1 bridge and bias/normalization diagnostics, then one controlled public-data fit comparison, and only then a compact causal context extension if the diagnostics support it. No method selection may use current-record truth; all predictions must be frozen before evaluator-private scoring.

The machine-readable record is `experiments/TRR-0004/status.json`.

## Historical A1 recipe checkpoint

The retained A1 state is `outputs/TRR-0002/blind/reconstructor_input/public_a1_lens.pt` (16,787,653 bytes; SHA256 `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`). Its saved metadata exposes only `hidden=0` and `corpus=alpaca`. The documented public recipe is sourced from the historical `attack.py`/`inv_common.py`, recipe document, explicit protocol, and closest machine-readable manifest recorded in `status.json`; that closest manifest is for a different raw-base lens and cannot establish exact provenance for the retained state.

The recipe uses `tatsu-lab/alpaca` (52,002 rows), a seed-7 random permutation selecting 1,200 sequences, 32-token minimum, 192-token maximum, and 125,571 collected positions. It renders instruction plus optional input with the Llama chat template and appends the output, tokenizes with `add_special_tokens=False`, and truncates to 192 IDs. The affine lens starts at identity/zero bias/scalar scale 3, normalizes the public embedding table, and uses full-vocabulary CE with AdamW (learning rate 1e-3, weight decay 0), batch 512, 3,000 steps, cosine annealing, optimizer seed 0. Exact retained row indices/order and measured fit duration remain unavailable; the historical cost is only documented as approximately two minutes.

The setup access incident is disclosed in `status.json`: a broad prior-output search surfaced one unrelated TRR-0002 evaluator-private `source_text` field in command output. It was not saved or used, and no TRR-0004 truth or record was touched.

## Access incident disclosure

A broad prior-output search accidentally surfaced one `source_text` field from an unrelated TRR-0002 evaluator-private JSON in command output. It was not saved, parsed, copied, used for reconstruction, selection, or evaluation, and no TRR-0004 record or truth was touched. Subsequent setup work is restricted to explicitly public metadata paths; the incident is recorded in `experiments/TRR-0004/status.json` and reported to root.

## Public activation preparation (implemented, not run)

The task-local preparation path is implemented by
`src/token_reconstruction/public_activation.py` and
`scripts/trr0004_prepare_public_activations.py`, with focused tests in
`tests/test_public_activation.py`. The intended command is create-only and
uses the pinned public Arrow cache, tokenizer/model snapshot, the existing
public Pile validation receipt, and `--batch-records 8`:

```text
timeout --signal=TERM 600s env HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:scripts .venv-trr0004/bin/python scripts/trr0004_prepare_public_activations.py --split-plan experiments/TRR-0004/alpaca_split_plan.json --dataset-arrow /home/alanz/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017/alpaca-train.arrow --dataset-info /home/alanz/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017/dataset_info.json --tokenizer /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --model /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --pile-receipt /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_validation_slice_v2/validation_slice_evidence.json --output-root outputs/TRR-0004/public_activation_v2 --device cuda --batch-records 8 --min-free-gpu-bytes 8589934592 --max-reserved-gpu-bytes 8589934592 --max-host-rss-bytes 17179869184
```

The planned train artifact is 1,200 records × 192 positions × 2,048 hidden
values in BF16 (943,718,400 bytes for activations), with token IDs, right-padding
attention masks, position IDs, and nested post-BOS selectors for 5,000 and
approximately 125,000 positions. The validation artifact is 24 records with
the same padded 192 geometry. The selectors refer to the current-token
activation at each post-BOS position; BOS is retained in the tensors but never
selected. `forward_full` receives the stored token IDs and uses the public
prefix's causal computation; the stored attention and position fields are
metadata/alignment contracts, while right-padding is semantically inert for
these causal prefixes and padded activation rows are zeroed after capture.

Before materializing the full split, the CLI hashes and verifies the actual
public model weight file against the pinned expected digest. It then qualifies
the largest declared geometry (fixed batch 8 × 192) by comparing its padded
capture with per-record unpadded reference captures through the same
`ContiguousPublicPrefix.forward_full`. The primary check changes only future pad
token IDs in the same [batch,time] shape and requires `torch.equal` on active
outputs. It separately records the actual batch-1 unpadded max absolute
difference, relative L2, and bit-exact result as a diagnostic; a mismatch is
labelled `non_equivalent_not_used` and never relaxed by a tolerance. A
fail-closed resource check runs after
model load, qualification, every capture batch, and final capture: preliminary
limits are 8 GiB minimum free GPU, 8 GiB maximum reserved GPU, and 16 GiB
maximum host RSS, with a 600-second process timeout. The conservative estimate
is about 6 GiB GPU peak and 10 GiB host peak, including the 2.47 GB public
weight file, BF16 activations, and validation copies. No activation preparation
or confirmation record has been run in this task yet.

The later confirmation split must use explicit public selection metadata for
any overlapping dataset and current fit/validation records. Exact retained
historical A1 row IDs are unavailable; that provenance uncertainty is recorded
and does not block a dataset-disjoint confirmation on the planned Pile/Finance
records. No private truth contents are used to construct this public split.

## First guarded preparation attempt (excluded)

The first create-only attempt used the frozen runner at commit
`32b042793a1bd7012ab0e2c0d9cbf5d463ea98c2` and exited before model load with
`ActivationPreparationError: tokenizer has no declared public padding fallback`.
The pinned tokenizer has `pad_token_id=None` and scalar `eos_token_id=128009`,
but its public vocabulary maps `<|end_of_text|>` to ID `128001`; the model
configuration also lists that ID among its EOS IDs. GPU peak allocation and
reservation were both zero, and target weights/private truth were not accessed.
The failure receipt and launcher logs are retained at
`outputs/TRR-0004/public_activation_v1/failure.json` and
`experiments/TRR-0004/evidence/public_activation_launch_v1/`.

The corrected loader accepts this fallback only when
`convert_tokens_to_ids("<|end_of_text|>") == 128001`, the ID is in the
`128256`-entry vocabulary, and `convert_ids_to_tokens(128001)` round-trips to
`<|end_of_text|>`. The tokenizer is not mutated. The local snapshot check and
mocked fallback test pass; root must commit this correction before the retry,
which uses the new create-only output root `public_activation_v2`.

## Successful public activation preparation (v2)

The corrected create-only run completed at frozen commit
`3a7206ec6440b3344b3c4db72b543c17152bfac7` using the task-local venv and
public-only inputs. The launcher receipt is
`experiments/TRR-0004/evidence/public_activation_launch_v2/launch.json`; its
stdout/stderr and the full preparation receipt are retained beside it. The
process completed in 9.764 seconds with no network access. Peak allocation was
2,472,678,400 bytes and peak reservation 2,480,930,816 bytes on the RTX 5080;
host RSS peaked at 6,173,032,448 bytes. All values passed the 8 GiB reserved
and 16 GiB RSS guards.

The required largest geometry qualification passed: the fixed batch-8 × 192
path changed 433 future padding token IDs in the same shape and observed
bit-exact active outputs (maximum absolute difference 0.0). The separate
unpadded batch-1 diagnostic was not used for capture and is recorded as
`non_equivalent_not_used`: maximum absolute difference 2.0, relative L2
0.0035975385, bit-exact false.

The public artifacts are in `outputs/TRR-0004/public_activation_v2/`:
`train_large_cut4.safetensors` (947,176,648 bytes; SHA256
`d1c78fcf1acc91b57d51355ee11f267bf4c12f1bc7d5160164b3b6ea11b45344`),
`train_large_records.json` (726,532 bytes; SHA256
`34e10ddc7c502b30822730b84c6f3a48b6285888b8ef92fc9e1aef65ecca29df`),
`validation_alpaca_cut4.safetensors` (18,945,712 bytes; SHA256
`9f0d051252ec4a2f9d48a48d7500049c4a10710244572e82d2d122bc7b0c3337`), and
`validation_alpaca_records.json` (12,943 bytes; SHA256
`b31cbdecbaf5b2617e8e20b3d32d7840316a7daf9e35ccf92870d15b189d190f`). The
train tensor geometry is 1,200 × 192 × 2,048 BF16 with 124,371 post-BOS
positions and a nested 5,000-position selector; validation is 24 × 192 ×
2,048 with 2,197 post-BOS positions. The preparation receipt confirms no
target weights, evaluator-private truth, or confirmation records were accessed.

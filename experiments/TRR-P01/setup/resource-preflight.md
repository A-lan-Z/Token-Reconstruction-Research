# TRR-P01 setup and resource preflight

Captured 2026-09-05T07:14:42Z in the isolated task worktree. This note records
setup and read-only resource checks only; no experiment job was launched.

## Isolation and provenance

- Worktree: `/tmp/trr-p01`
- Branch: `task/TRR-P01`
- HEAD: `6b618760f50055dc5c8a62e830ab7a9761190cfe`
- Requested parent: `6b618760f50055dc5c8a62e830ab7a9761190cfe`
- HEAD parent: `88af8480060c323247aaad69d2848fa0a9261819`
- `git merge-base --is-ancestor 6b618760f50055dc5c8a62e830ab7a9761190cfe HEAD`: exit 0
- Configured remote: `origin https://github.com/A-lan-Z/Token-Reconstruction-Research.git`
- Incoming packet source: `/mnt/c/Users/alanz/.codex/attachments/5a641df9-edd6-4381-aa32-818e964a7ba4/pasted-text.txt`
- Preserved packet: `coordination/requests/TRR-P01.md`
- Packet SHA-256 (source and copy): `27d6a27f935a1b84e2faf8a12e8d58cd87331e5d0e8c131502025fcaf6d697d2`

The parent checkout was on `task/TRR-0003` with unrelated untracked TRR-0003
files. It was not switched or edited. No TRR-0003 output files were opened.

## Runtime and host

- WSL distribution: Ubuntu 24.04.4 LTS (`ID=ubuntu`, `VERSION_ID=24.04`)
- Kernel: WSL2 `6.6.87.2-microsoft-standard-WSL2`, x86_64
- Windows host version reported by PowerShell: `10.0.26200.0`
- Windows UNC resolution: PowerShell `Test-Path -LiteralPath \\wsl.localhost\Ubuntu\tmp\trr-p01` returned `True`; `Get-Item` returned the same UNC path. `cmd.exe` also returned `EXISTS`.
- CPU: AMD Ryzen 9 9950X3D, 16 cores / 32 logical CPUs
- RAM: 30 GiB total, 25 GiB available at capture; 8 GiB swap free
- Filesystem: 1,007 GiB volume, 295 GiB free at capture (70% used)
- Python: `/usr/bin/python3`, Python 3.12.3; pip 24.0
- Python search path includes `/home/alanz/.local/lib/python3.12/site-packages`
- Installed versions: torch 2.10.0, transformers 5.3.0, safetensors 0.7.0,
  datasets 4.8.3, numpy 1.26.4, pytest 8.4.2
- The version record in `environment/known-good.txt` reports torch
  `2.10.0+cu128` and CUDA `12.8`; no CUDA allocation or model load was done
  during this preflight.

## GPU reservation

The host exposes one device: GPU 0, NVIDIA GeForce RTX 5080, 16,303 MiB total.
At capture `nvidia-smi` reported 6,393 MiB used, 9,585 MiB free, and 95% GPU
utilization. Read-only process inspection showed an active guarded TRR-0003 CUDA
training process. The device is reserved by root coordination; this agent did
not allocate CUDA memory, launch work, or stop any process. There is no OS lock
installed. Any future GPU run must wait for an explicit slot and repeat the
fail-closed resource guard and allocation probe.

## Published baseline and public assets

The pinned public model is available locally at:

`/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6`

Its public metadata identifies a LlamaForCausalLM with 16 layers, hidden size
2048, vocabulary size 128,256, BOS token 128000, and bfloat16 weights. The
snapshot has a 2,471,645,608-byte `model.safetensors`, 9,085,657-byte
`tokenizer.json`, and the following checked hashes:

- `config.json`: `2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb`
- `tokenizer_config.json`: `9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e`
- revision: `9213176726f574b556790deb65791e0c5aa438b6`

The published heavy full-SFT snapshot is also cached at
`/home/alanz/.cache/huggingface/hub/models--Vikhrmodels--Vikhr-Llama-3.2-1B-Instruct/snapshots/7fa9d06a59246629244cdd3b6b92e4fc756baa0f`
with a 2,471,645,464-byte model file. It is not required for the initial
matched-model pilot and was not loaded.

The pinned public Pile-10k revision is cached at:

`/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa`

The cache contains `pile-10k-train.arrow` (61,270,696 bytes) and
`dataset_info.json` (736 bytes). The public content fingerprint blob hashes to
`a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31`, matching
`resources/public_resources.json`.

The historical public Alpaca-fitted lens comparator is available at:

`/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/inversion_20260730/out/lens_alpaca.pt`

It is 16,787,653 bytes and hashes to
`33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`. It is a
published historical comparator only; the no-fit prototype arm must not use it.

## Frozen observation and comparator paths

The following previously published, non-TRR-0003 assets are present locally and
can be used as diagnostics if their access conditions are retained:

- Clean canonical TRR-0001-R1 reconstructor input:
  `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0001-R1/clean/frozen_bundle/reconstructor_input`
  (cut-4 inverse, sanitized metadata, and matched/target-LoRA observations).
  The cut-4 inverse is 16,785,688 bytes and hashes to
  `9e2487f85057748130bf87b2aad0a883f3c36dfc052eefd83c0f5c35497a24e3`.
- Historical TRR-0002 bridge input:
  `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/historical-input-target-bridge/reconstructor_input`
  with `config.json` (22,369 bytes) and `observations.safetensors`
  (201,573,552 bytes). The observation archive hashes to
  `a40b148c2232fa599a7c3b810a556b52c30c84b98efb8b935d121dd7424a71c8`.
- Published historical bridge predictions:
  `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/historical-input-target-bridge/predictions`
  (checkpoint-identity and historical-Alpaca A1 controls).
- Published strict-BOS baseline method state:
  `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0001/frozen_bundle`
  (cut-4 state hash matches the inverse hash above).

These paths are recorded for asset availability and provenance. The pilot must
not open private truth sidecars or use target truth while constructing or
freezing predictions.

## Memory estimate and practical path

For the full static boundary-prototype table at the pinned geometry,
128,256 vocabulary entries × 2,048 hidden values require exactly 525,336,576
bytes (501 MiB) in bfloat16 or 1,002 MiB in float32, before temporary batches
and normalization buffers. The public model file is about 2.47 GB on disk. CPU
RAM has enough nominal capacity for a model plus a compact table, but the full
prefix pass over 128,256 two-token sequences is a substantial CPU workload;
use a small batched smoke first and retain timing and peak-RSS evidence. GPU
execution is currently unavailable by reservation. No heavy run or table build
was launched in this setup task.

The safe next step is a CPU-only geometry/cache smoke and a bounded prototype
batch on matched public observations. Before any representative full-vocabulary
or target-shift run, root coordination must release a GPU slot, estimate the
largest tensor geometry, qualify it with a live resource guard, and record
output equivalence for any batching workaround.

## Publication authentication preflight

A normal Windows Git credential-manager path was tested read-only on 2026-09-05.
The Windows Git executable was `/mnt/c/Program Files/Git/cmd/git.exe`; its
common repository directory was addressed through the verified Ubuntu WSL UNC
path because the worktree `.git` file points at a Linux worktree-admin path.
The per-process environment was `GIT_TERMINAL_PROMPT=0` and
`GCM_INTERACTIVE=Never`; no credential contents were read or printed.

The first dry-run used `HEAD:refs/heads/task/TRR-P01` and returned exit 0, but
that source form is not accepted as publication provenance when a common Git
`HEAD` can belong to another task. A second dry-run used the explicit source
and destination ref `refs/heads/task/TRR-P01:refs/heads/task/TRR-P01`; it also
returned exit 0 with no push side effect. At that check the local task ref
resolved to commit `0f1b762eef2fafbc555818ecfe451909c821e98a`. Linux `gh` was
not installed and the Linux HTTPS helper could not obtain a username; the
normal Windows Git path is therefore the available publication route. No
branch, index, checkout, environment, or remote state was changed by either
dry-run. Any later push must recheck the source ref and use the explicit task
ref, never generic `HEAD`.

## Verified post-BOS construction diagnostic

The root-authorized CPU-only diagnostic ran after code commit
`0f1b762eef2fafbc555818ecfe451909c821e98a` with CUDA hidden:

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/trr_p01 python3 scripts/trr_p01/check_post_bos.py --build-root experiments/TRR-P01/runtime/cpu-table-20260905 --plan experiments/TRR-P01/pilot_plan.json --output-root experiments/TRR-P01/runtime/post-bos-verified-20260905 --implementation-commit 0f1b762eef2fafbc555818ecfe451909c821e98a
```

The public prediction artifact was frozen before reading expected probe labels;
the receipt bound the plan and qualification-report bytes, and the post-freeze
check confirmed that both files were unchanged and carried the same ordered
256 probe IDs. The result was `PASS`: cosine and L2 each identified 256/256
probes, with zero ties and zero prediction collisions. `truth_opened=false`,
no private truth was read, no target condition was exposed, and no CUDA memory
was allocated. This is a construction/identity diagnostic, not a comparative
panel score or an uncontended timing claim.

The recorded artifacts are under
`experiments/TRR-P01/runtime/post-bos-verified-20260905/`:

- `preflight.json`: guard PASS, 23,619,223,552 bytes available at preflight,
  762,465,188 bytes required, and 525,337,024-byte table input.
- `post_bos_predictions.safetensors`: SHA-256
  `7ab0cdcec857968072049216fa93c90f9f4f19be1ef331c4b23b25d83b08d83e`.
- `post_bos_freeze.json`: SHA-256
  `5ed08c85e08dd606dd6070f994a0a6be0edcb9a014064a8a620b8e7b0e7ac096`.
- `post_bos_identity.json`: SHA-256
  `64091cd037d06f39287c1a8ecb63a161fd461e8b2b481a57d231426a5813faa8`.

The run took 2.461590155999147 seconds in total: 0.17310120899855974 seconds
table/query load, 1.5966075810010807 seconds lookup, and 0.4608187510002608
seconds post-freeze comparison. Process maximum RSS was 2,412,810,240 bytes.
The pinned table hash was
`51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3`; the
qualification report, chosen output, and plan hashes were respectively
`1c4f7ce680c1570c984eb8c881ff4ee06b84672fe83d7372e329746d6a0dedaf`,
`117f5cc826c55d5f97bcc17e04e979f1b8ea59e12f8dbf5d7b474be7271bd375`, and
`2e68b8ce7514c9f8338d47f8c3cc56f957259453b057a4c5a3cef1273268bfce`.
The diagnostic source file hash was
`e24b1f0570e69bee092045d5fe7f963ea2e7201c4ce8263bd48b6317ef73bea1`.

## CPU method-cell qualification

The largest representative method cell was qualified on CPU under implementation
commit `9fb635a7f66866da05a08cb6084da7b1704f13a3` with CUDA hidden. The exact
recorded command is in
`experiments/TRR-P01/review/qualifier-cpu-20260905.command.txt`:

```text
timeout --foreground 900s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src:scripts/trr_p01 python3 scripts/trr_p01/qualify_methods.py --plan experiments/TRR-P01/pilot_plan.json --build-root experiments/TRR-P01/runtime/cpu-table-20260905 --historical-lens /home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt --output-root experiments/TRR-P01/runtime/method-qualification-cpu-20260905 --implementation-commit 9fb635a7f66866da05a08cb6084da7b1704f13a3
```

The cell passed all CPU guards and wrote
`experiments/TRR-P01/runtime/method-qualification-cpu-20260905/method_qualification.safetensors`.
It used eight records × a 256-candidate budget at prefix length 39, for 2,048
candidate simulations and 2,048 candidate cache commits. The cell also made
312 persistent prefix-cache commits, eight reference-token-220 evaluations and
cache commits, and 2,688 public-prefix input-token evaluations. Candidate
simulation took 0.4646248879998893 seconds, the reference probe took
0.017231914000149118 seconds, and full-vocabulary cosine/L2 lookup for eight
queries took 0.903067810999346 seconds. The complete cell took
5.413899522000065 seconds, with peak RSS 5,475,844 KiB (5,607,264,256 bytes; the raw
KiB value is retained in the evidence). The pre/post-cell guards required
5,295,880,560 bytes and observed at least 21,353,287,680 bytes free; all guard
statuses were `PASS`.

The qualification evidence reports exact equality for the cached-prefix versus
full-prefix public probe (`maximum absolute difference = 0.0`) and
`truth_opened=false`. This cell validates resource geometry and the fixed
historical A1+A2 comparator policy; it is not a panel accuracy result, and its
CPU timing is subject to shared-host workload. The method artifact hashes are
`0196d7cd34342ee5ed7fffb9dfe0ea83bb3f50f78e1c58ef03cb2c074deed88a` for the
output, `1592d7aa262eb9c27584a60f78e4cf431d6e31fd62cc90b94ccafa63f4187460` for
the evidence, and `d900cb90497f2254508b445d6d620b3ba0263232bc437437303d7c0501830435`
for the preflight. The command and log hashes are
`d7638a642d12792bb2822e50606bb33f2f79ebc728ce0fa12a6c2b06d2c8eee3` and
`aebdf58f599bcc46edb60e1424c00861616d8ae8726af8875f06a1a6d0195b81`.

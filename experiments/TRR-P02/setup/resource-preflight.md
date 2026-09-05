# TRR-P02 setup and read-only resource preflight

Captured 2026-09-05T11:06:09Z in the isolated `/tmp/trr-p02` worktree. This
note records setup and read-only checks only. No model was loaded, no prototype
table was built, and no reconstruction or diagnostic job was launched.

## Isolation and provenance

- Worktree: `/tmp/trr-p02`
- Branch: `task/TRR-P02`
- Starting and current commit at setup: `e3e8a1de020598fb68c1ed8b64c0e155823817f5`
- Parent snapshot: P01 publication head on `task/TRR-P01`
- Incoming packet source: `/mnt/c/Users/alanz/.codex/attachments/1418d7e2-e1bb-4a75-80d6-334db93acb75/pasted-text.txt`
- Preserved packet: `coordination/requests/TRR-P02.md`
- Packet SHA-256 (source and copy): `e16db54941ba85dd7f9f9e930578e9412ee18d348b1dc4c8746a9a25e103cad5`
- Common checkout was not switched or edited. No merge, rebase, global
  coordination update, active-method registry update, TRR-0004 output access,
  or authentication work was performed.

## Runtime and host

- WSL distribution: Ubuntu 24.04.4 LTS
- Kernel: `6.6.87.2-microsoft-standard-WSL2`, x86_64
- CPU: AMD Ryzen 9 9950X3D, 16 physical cores / 32 logical CPUs
- RAM at capture: 30 GiB total, 18.2 GiB free, 21.7 GiB available
- Swap at capture: 8.0 GiB total, 8.0 GiB free
- Disk at capture: 296 GiB available on `/`
- Python: 3.12.3 (`/usr/bin/python3`)
- Packages: torch `2.10.0+cu128`, transformers `5.3.0`, safetensors `0.7.0`,
  datasets `4.8.3`, numpy `1.26.4`, pytest `8.4.2`
- `torch.cuda.is_available()`: `True` (availability check only)

## Shared GPU and live-process check

At capture, `nvidia-smi` reported device 0 as NVIDIA GeForce RTX 5080, 16,303
MiB total, 2,883 MiB used, 13,095 MiB free, 11% utilization, 51 C, P3. The
other workstream subsequently reported that it was about to use the shared GPU
and allows at most eight CPU workers and 16 GiB process RSS. Accordingly, this
task launched no GPU allocation and will hold all model/table work until an
exclusive CPU diagnostic window is granted. A read-only process scan found no
active TRR, reconstruction, torch, or model worker; only the host services and
the scan itself were visible.

The P01 publication recorded a public prototype-table build peak of exactly
3,721,547,776 bytes (3.47 GiB), and a grouped final full-40-slot panel peak of
about 5.24 million KiB (about 5.00 GiB). Those are historical reference
points, not a qualification for this task. Short public probes remain
resource-unqualified here and require a fresh guarded window before execution.

## Public cached assets available read-only

The following assets were checked by path and metadata only. Their contents are
public resources or already published diagnostic controls. No target weights,
private truth tensors, source plaintext, or evaluator-private directories were
opened.

| role | identity and revision | path | observed status / hash |
| --- | --- | --- | --- |
| public matched model | `meta-llama/Llama-3.2-1B-Instruct`, `9213176726f574b556790deb65791e0c5aa438b6` | `/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6` | metadata present; `config.json` 877 B, SHA-256 `2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb`; `tokenizer_config.json` 54,528 B, SHA-256 `9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e` |
| public auxiliary data | `NeelNanda/pile-10k`, `127bfedcd5047750df5ccf3a12979a47bfa0bafa` | `/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa` | `dataset_info.json` 736 B, SHA-256 `6f76b8c5908b60192866afdcf2ff773bb877ae7c8240d9007e841633af21ba0e`; 10,000 public rows |
| historical fitted-lens control | public Alpaca affine lens, published comparator | `/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/inversion_20260730/out/lens_alpaca.pt` | 16,787,653 B, SHA-256 `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`; comparator-only, no-fit arm may not use it as a learned asset |
| P01 public prototype table | cut-4 boundary prototypes, model-native BF16 | `experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors` | published artifact, 525,337,024 B, SHA-256 `51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3`; reuse read-only only |
| P01 public identity control | 256-token BOS construction diagnostic | `experiments/TRR-P01/runtime/post-bos-verified-20260905/post_bos_identity.json` | published artifact SHA-256 `64091cd037d06f39287c1a8ecb63a161fd461e8b2b481a57d231426a5813faa8`; no truth opened |

Published P01 clean and historical observation bundles are available as
diagnostic inputs by their existing paths, but are not opened during setup.
Any newly generated public component IDs will be labelled diagnostic pairs and
will not be treated as reconstruction claims.

## Read-only commands

The preflight consisted of `date -u`, `uname`/`/etc/os-release`, `lscpu`,
`free`/`/proc/meminfo`, `nvidia-smi --query-gpu`, `ps`, `df`, Python package
version and CUDA-availability imports, `git status`/`git worktree list`, and
file `stat`/SHA-256 checks for the public metadata above. No command loaded a
checkpoint or wrote an experiment result.

## Next gate

Wait for an exclusive short CPU diagnostic window before any public model
forward pass. At that point, run only the smallest declared probes needed to
check offset sign, position alignment, cache transitions, and shared-offset
geometry, with a fail-closed resource guard and task-local receipts.

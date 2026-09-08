# TRR-0012 durable fixed-readout package contract v1

Status: **DRAFT_PRE_SELECTION**. This record freezes the deployment shape and
smoke inputs before the rebuilt B0/B1 fits. It is not a completed package and
does not authorize scientific evaluation. The actual selected state bytes,
post-selection prediction IDs, and the Agent2 restore receipt remain required.

## Scope and identity

The task is one new-weight replication of the two fixed public-readout methods:
current-bank B0 (`current_fixed`) and expanded-bank B1 (`expanded_fixed`). The
serialized decoder remains the source-bound `ResidualMLPPositionwiseDecoder`
with hidden size 2048, vocabulary size 128256, context width 128, bottleneck
512, and seed 4005. The strict package loader is
`code/trr0010_p09_fixed_loader.py`; its serialized state method id remains
`continued_fixed_readout`, while the package-level roles are `current_fixed`
and `expanded_fixed`.

The old TRR-0011 state hashes are absent and are not used. The common starting
decoder identity is recorded as recipe provenance only:
`5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`,
29,390,492 bytes. The selected rebuilt states must receive new immutable model
ids and new file/tensor identities.

## Portable inventory

The primary package destination is
`/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0012/outputs/TRR-0012/model-package`. Root will create the secondary copy at
`/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012` only after measuring the completed primary package and checking
that at least 10,737,418,240 bytes remain on the Windows filesystem. The
recorded devices are 2096 and 68. No symlink or hardlink may point back to a
training asset. The package cap is 2,147,483,648 bytes. The current package skeleton is 2,264,594 bytes; adding the exact E size
and two prior 29,390,492-byte state-size estimates gives a planning estimate
of 1,111,719,066 bytes before final sidecars, receipts, and expected outputs. The final state bytes and all metadata must be measured. The final size
must be measured from actual files before copying.

| Role | Package-relative path | Planned identity |
| --- | --- | --- |
| B0 selected state | `states/current_fixed.safetensors` | new bytes/SHA after selection; prior size is only an estimate |
| B1 selected state | `states/expanded_fixed.safetensors` | new bytes/SHA after selection; prior size is only an estimate |
| public normalized E | `readout/public_normalized_embeddings.safetensors` | 1,050,673,488 bytes; SHA `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1` |
| strict loader | `code/trr0010_p09_fixed_loader.py` | 8,174 bytes; SHA `1225ebc9871aadae4a287a8a5c5a80c5d9903fc86ec5f83ae0bfdae8c9e94a94` |
| package CLI | `code/trr0012_package.py` | 59,473 bytes; SHA `9083926a0c369e3ca4361108bea043fbda79f619ff0fca2ce76b780d11bf0109` |
| decoder source | `code/token_reconstruction/trr0007_positionwise.py` | 18,152 bytes; SHA `89fcd036a57407c8c49294aa5fd15ccef46f02b6fabcc35435d097f1213de485` |
| decoder import closure | `code/token_reconstruction/access.py`, `io.py`, `public_prefix.py` | exact hashes in `contract_v1.json`; required because the vendored namespace imports them |
| decoder dependency | `code/token_reconstruction/trr0005_joint_decoder.py` | 55,256 bytes; SHA `9cab3cf5ad664c37d91bcd90fe1852df71da25800ddf37291a57000908d3d38f` |
| smoke input | `smoke/public_base_first2.safetensors` | 2,102,632 bytes; SHA `652a67744f2dabe2ca53198f1b7fddde6cb36d414280a1a0d95d866d1e5cc840` |
| tensor identities | `identity/*.safetensors.json` | generated from actual state/E tensors at freeze |
| selection receipt | `receipts/selection_complete_before_smoke.json` | generated after ordinary selection, before smoke outputs |
| expected smoke output | `smoke/expected_predictions.safetensors` | generated after selection with the same package loader |
| final manifest/config | `package_manifest.json`, `config/frozen_config.json` | create-only final bindings |

The smoke input fixture is bundled and contains only public H, mask, and
position slices. It contains no source text, target labels, token IDs, or
truth. Its source observation files were hash-bound before fitting; those
source files are not deployment dependencies.

The bundled consumer command is:

```text
python code/trr0012_package.py predict --package-root . --observations <observations> --output <create-only-output> --device <cpu-or-cuda>
```

It requires `receipts/selection_complete_before_smoke.json`; the final
`package_manifest.json` follows `package_manifest_template_v1.json`, with both
methods' loader kwargs and selected state bindings filled with actual hashes. It
accepts either standard tensors `activations`, `attention_mask`, `position_ids`
with shapes
`[N,128,2048]`, `[N,128]`, `[N,128]`, or the bundled domain-prefixed smoke
fixture, and emits one safetensors file with int64 keys `current_fixed` and
`expanded_fixed`, each `[N,128]`, plus a `.receipt.json` record. The receipt
binds the observation file, both state hashes, both full prediction tensor
digests, per-record digests, and the explicit truth boundary. It also records
the actual loaded readout/state paths and file bindings, local vendored module
paths, and Python/Torch/safetensors/numpy versions. Each row is computed
through the same strict vendored loader and FP32 decoder path.

## Frozen smoke order

The four records are fixed in this order for both methods:

1. `finance/public_base/000` — `Josephgflowers/Finance-Instruct-500k/train@583a98fb0ec14d904e9423b671d9d0fea88891b6:row-012278` — public record SHA `57fa99cb426d5d0f6d47881bd377a08ac25a69696494652279e6fc7725427ea5`
2. `finance/public_base/001` — `Josephgflowers/Finance-Instruct-500k/train@583a98fb0ec14d904e9423b671d9d0fea88891b6:row-014057` — public record SHA `5b7929fd81f0ff5fbec708e1841a393babe33e58ece8efa4cfdfee7b71314f66`
3. `pile/public_base/000` — `NeelNanda/pile-10k/train@127bfedcd5047750df5ccf3a12979a47bfa0bafa:row-007721` — public record SHA `d01a69e865d706dfc667e10b624021d10279087f63a9a4f8655e3f448063a5e6`
4. `pile/public_base/001` — `NeelNanda/pile-10k/train@127bfedcd5047750df5ccf3a12979a47bfa0bafa:row-008703` — public record SHA `993a8875128088946c08a1be5c90ea48c32e9487782ecd0c7bc6567dc1eca399`

The fixture geometry is BF16 H `[4,128,2048]` with all-valid `[4,128]`
attention masks and position IDs `[4,128]`; inference casts each record to
FP32, prepends BOS token 128000, and emits IDs `[4,128]` with 127 predicted
post-BOS positions. The per-record canonical tensor digests are in
`contract_v1.json` and the extraction receipt.

## Fit and selection bindings

The fit prerequisites are bound by the task inventory: model/tokenizer
snapshot revision `9213176726f574b556790deb65791e0c5aa438b6`, persistent B0
payload SHA `a55814759dfa9d2567587935063fc49e44d8bff949c50014793deb982ebdf35d`,
and ordered B1 ledger SHA
`48b6656845a18428e1fdd54c194928538cd9aa87dfb010b5238f11df3c33fbb0`. The
B1 runtime payload is explicitly marked lost and must be rebuilt under the
separate preparation gate; none of these training assets enters the consumer
package.

The documented recipe uses full fit context width 192, natural validation
width 128, record batch 8, position budget 512, schedule seed 4010, AdamW
(`foreach=False`), cosine decay, base learning rate 2e-4, weight decay 0,
and gradient clip norm 1.0. The checkpoint grid is the shared seven-point grid
`[0,1000,2000,4000,8000,12000,13000]` for both arms. Select the earliest
strict maximum of domain-balanced public validation token accuracy using
development/selection data only. New bank, fit, schedule, runner, selected
step, and selected state hashes must be filled from the new run receipts.

Selection receipt fields are mandatory and explicit:
`complete_before_smoke=true`, `smoke_used_for_selection=false`,
`independent_evaluation_truth_opened=false`,
`checkpoint_reselection_after_smoke=false`, and a boolean
`development_labels_used`. Expected IDs are recorded only after ordinary
checkpoint selection, never used to select or reselect a checkpoint.

## Restore gate

The deployment manifest at `package_manifest.json` is a consumer descriptor
without a self-hash field. Its byte/SHA binding is held by the external Agent2
restore descriptor, which prevents a recursive self-hash. A bounded synthetic CPU probe passed against the vendored loader: 15 exact
tensor keys, state schema, and all metadata bindings restored successfully;
the synthetic file was 29,390,284 bytes, max RSS was 666,864 KiB, and no E load,
full-vocabulary forward, CUDA, or truth access occurred. The receipt is
`package/loader_compatibility_receipt_final.json`.

The serialized state
metadata remains `token-reconstruction.trr-p09-fixed-state.v1` with method id
`continued_fixed_readout` because that is the strict loader contract; the new
replication identities live in the selection receipt and consumer manifest.

The Agent2-compatible descriptor template is embedded in
`contract_v1.json` under `restore_template` with schema
`token-reconstruction.trr-p11-restore-gate.v1`. It requires two real copies
of each state, E, code, config, selection receipt, tensor identity sidecar,
smoke input, and expected output. Each copy is bound by relative path, byte
count, and SHA-256; state/E sidecars additionally bind sorted tensor keys,
dtypes, shapes, and canonical tensor digests. The clean runtime is
`/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-P11/restore-runtime/TRR-0012`, outside this training worktree and temporary storage.

The final sequence is: finish ordinary selection; persist both actual states;
assemble and measure the primary package; generate the expected smoke IDs;
freeze the primary package create-only; have root copy it to Windows; then
Agent2 retrieves into an empty clean runtime and checks file hashes, tensor
identities, and both methods' `[4,128]` predictions. A manifest or pointer
without retrievable state bytes is incomplete. Scientific evaluation remains
blocked until that independent restore succeeds.

## Missing values before freeze

The contract intentionally leaves the following unresolved: new selected state
bytes/SHA and tensor identities; selected steps/model IDs; regenerated bank,
fit, schedule, and runner hashes; actual E copy identity sidecar; selection
receipt; expected prediction IDs/digests; secondary byte-equality receipt; and
the clean Agent2 restore result. Filling these values from old TRR-0011
metadata would violate the recovery amendment.

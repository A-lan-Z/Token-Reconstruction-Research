# TRR-0010 A1+A2 availability and cost note

**Status:** corrected focused availability record, written 2026-09-07. This note
supersedes the A1/A2 portions of `evaluation_proposal.md`; that earlier file is
preserved unchanged as proposal history and has no authority. This note records
published repository evidence and one aggregate-only public capacity scan. It
does not select sources, run an A1+A2 reconstruction, open truth, or create
predictions.

## Correct comparator identity

TRR-0010 should carry **one composite A1+A2 comparator**, not separate A1 and
A2 scientific arms. The historical active method is
`frozen_a1_a2_k256`, also recorded as
`a1_a2_exhaustive_configuration_winner` in TRR-0002. Its frozen operation is:

1. use the retained public Alpaca lens as the A1 proposer;
2. propose at most 512 candidates in chunks of 256;
3. use the public causal prefix (untouched public Llama layers 0--3) to score
   the proposed candidates by direct cosine, with fixed selector budget K=256;
4. commit the final K=256 winner at every position.

The policy is `a1a2_43ea0bb737bc075531ca`, with `schedule=[256]`,
`score_rule=direct_cosine`, `fast_path_id=off`, no adaptive routing, no
candidate-group centering, no abstention, and terminal action
`commit_last_winner`. The causal run makes zero target-prefix calls; after BOS,
its prefix is its own reconstructed prefix. The historical proposal budget is
512 and its proposal chunk is 256; the actual committed inference operation is
fixed top-256. A pure `historical_alpaca_a1` row in older matrices is
provenance/control evidence and must not be relabelled as the TRR-0010 A1+A2
comparator.

The governing winner and policy binding is
`experiments/TRR-0002/configuration-search/causal-selection/winner.json`
(18,525 bytes, SHA-256
`a75a5220647b0dea019cabb09be7c99a82f0d8142bf5476ffff6be5099fcb4f3`). The
published protocol calls this fixed K=256 direct-cosine policy the
accuracy-first A1+A2 winner. TRR-0007 later used the same mechanism as a
bounded port named `bounded_a1_a2_k256_p0`; that port is useful fixture evidence
but is not a replacement for the native frozen `frozen_a1_a2_k256` identity.

The TRR-0010 unchanged shared starting reference is the TRR-0009 continued
fixed-readout checkpoint at selected step 400:

- path: `experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors`;
- bytes: 29,390,492;
- SHA-256: `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`.

The TRR-0007 residual state with SHA-256 beginning `2a44a91b...` is an older
published state and is not the TRR-0010 unchanged baseline. Any gap-closure
comparison is therefore **TRR-0010 unchanged shared start versus the single
composite A1+A2 comparator**.

## Source and public-asset proof

The exact native comparator binding appears in
`experiments/TRR-0003/evidence/timing/comparator_matrix_v2_bindings.json`
(5,501 bytes, SHA-256
`9bb50bb7e723631249f766c657a1d44dc6f0ca199d97db46c789c8abc7448678`). It
records code commit `b169134f7b97eb4447ad92369fb08909aeea25b7`, method rule
“public Alpaca A1 top-256 plus fixed public-prefix direct-cosine K256”, and the
following source/state bindings:

| role | published binding |
| --- | --- |
| A1+A2 policy implementation | `src/token_reconstruction/a1a2_configuration_search.py`, 29,969 bytes, SHA-256 `608bead6291353cacaa700a85f3dd619fb3261375e6fd5ba19cbc43fd738315b` |
| component dispatch | `src/token_reconstruction/component_crossover.py`, 20,708 bytes, SHA-256 `2cd03fda1f29f30d0a92246252ba2d2d89d3890abc6739ae0c8bae0a5c105b7d` |
| native comparator driver | `scripts/trr0003_footing_compare.py`, 44,976 bytes, SHA-256 `bb3690a481014f3ea12b411a3ec6eb58d7c888ef6c4135f9e03908027857f98e` |
| public prefix implementation | `src/token_reconstruction/public_prefix.py`, 8,297 bytes, SHA-256 `b9dca1d8d56c7c07413015aea4bde79e8ceac827c32ee9bc0d1a236ea1dd35f6` |
| retained public A1 lens | `experiments/TRR-0004/evidence/comparators/public_a1_lens.pt`, 16,787,653 bytes, SHA-256 `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742` |
| public reference helper | `round001_teacher.py`, 31,291 bytes, SHA-256 `10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd` |

The public runtime assets are bound in the TRR-0004 method freeze
(`experiments/TRR-0004/fresh_confirmation_method_freeze.json`, 9,306 bytes,
SHA-256 `df94093504693163c90775d2c4c9ffd23c63b16d19a52a2137d805a8c16154bc`):

- normalized public embedding table: 1,050,673,488 bytes, SHA-256
  `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`;
- public Llama prefix checkpoint: 2,471,645,608 bytes, SHA-256
  `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`;
- public prefix configuration: 877 bytes, SHA-256
  `2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb`;
- model: `meta-llama/Llama-3.2-1B-Instruct`, revision
  `9213176726f574b556790deb65791e0c5aa438b6`, BF16 with SDPA, public layers
  0, 1, 2, and 3.

The prefix checkpoint/configuration and embedding table are public runtime
resources. The retained A1 lens is a public-data-fitted asset. No private
adaptation state, target prefix, target labels, or sealed workspace asset is
part of this availability record.

The public model/prefix and lens binding was independently carried into
TRR-0005 and TRR-0007. TRR-0007's method-freeze anchor
(`experiments/TRR-0007/method_freeze.json`, 22,602 bytes, SHA-256
`301248af5cbdf95388ec1d39b42b00529227e4e3aa72613646dcd15a1f0dcfe9`) records
`proposal_k=512`, `proposal_chunk=256`, `fixed_k=256`, and the same lens hash.
Its adapter is explicitly labelled a legacy CPU-normalized-E numerical port;
that qualification is separate from the native comparator evidence below.

## Published fixture and measured cost

TRR-0002 provides the native configuration-search cost. On its clean Pile
64x40 setup it recorded 638,976 logical and executed candidate simulations,
15.30155777 seconds compute, 15.041352284 seconds selection, and
0.260205486 seconds proposal time. On historical Finance 128x128 it recorded
3,581,440 simulations, 140.315079788 seconds compute, 139.796414260 seconds
selection, and 0.518665528 seconds proposal time. The corresponding canonical
result is `experiments/TRR-0002/configuration-search/canonical/result.json`
(81,772 bytes, SHA-256
`271a4fa3bd4a093b6c4373ad9c704ca3ae74a8cf3712ba91a7f0fe4821f079f1`). The
fresh blind 64x40 receipt recorded 638,976 simulations, 15.112770003 seconds
method compute, 14.770360958 seconds selection, and 0.334138712 seconds
proposal time; it is `experiments/TRR-0002/configuration-search/fresh-blind/fresh-blind-result.json`
(8,317 bytes, SHA-256
`7f7838a774a03aecd49e4bcea5fae8284c0c63d81974ae3b7d214a2729841ac6`).

The native TRR-0003 comparator matrix gives the most compact per-cell fixture:
its panel is `experiments/TRR-0003/footing/panel.json` (88,693 bytes, SHA-256
`d1810330f53ebb7e149be45b8c07414e09d30b31b68c7c09d24df3854fec7333`), with
eight records per cell and batch size four. The four `frozen_a1_a2_k256`
records were:

| cell | elapsed seconds | candidate simulations | public-prefix calls | peak CUDA allocated / reserved | process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pile/public_base | 2.878535 | 79,872 | 392 | 2.532 / 5.385 GB | 3.393 GiB |
| Pile/public_lora_2601 | 2.509681 | 79,872 | 392 | 2.533 / 5.388 GB | 3.393 GiB |
| Finance/public_base | 10.212885 | 195,072 | 1,018 | 3.361 / 5.388 GB | 3.393 GiB |
| Finance/public_lora_2601 | 9.639186 | 195,072 | 1,018 | 3.361 / 5.388 GB | 3.393 GiB |

These are steady matrix timings; the separate TRR-0003 qualification cell
measured 10.388176 seconds at Finance/public_base and verified batch-4 output
against a batch-8 reference before timing.

TRR-0005 supplies a larger native fixture with 128 records per cell, batch
size one, one warmup and one measured call per record, synchronized after each
call, and exact warmup/measured ID equality. The four-cell timing artifact is
`experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export/timings.json`
(228,291 bytes, SHA-256
`c66c0f998e1afc3eef4fa43c3c35ec4128f16a9b4c63864f49b59381687f5ce1`). For
`frozen_a1_a2_k256`, the total warmup-plus-measured intervals were:

| cell | total timed interval | measured sum | warmup sum | load | candidate simulations | peak allocated / reserved | process RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pile/public_base | 262.310331 s | 130.725606 s | 131.584725 s | 9.236083 s | 8,323,072 | 2.397 / 3.546 GB | 5.816 GB |
| Pile/public_lora_2601 | 262.577532 s | 131.288910 s | 131.288622 s | 9.236083 s | 8,323,072 | 2.397 / 3.546 GB | 5.816 GB |
| Finance/public_base | 264.846509 s | 132.332180 s | 132.514328 s | 9.236083 s | 8,323,072 | 2.397 / 3.546 GB | 5.816 GB |
| Finance/public_lora_2601 | 266.647320 s | 133.135502 s | 133.511819 s | 9.236083 s | 8,323,072 | 2.397 / 3.546 GB | 5.816 GB |

Each 128-record cell used 65,280 public-prefix calls and 32,768 prefix-commit
tokens. Candidate arrays were output-only and not persisted. The historical
A1 fit duration for the retained lens is not machine-recorded; the provenance
audit reports only a qualitative historical estimate of about two minutes on
public data. TRR-0010 must therefore report zero new A1 preparation work for
reusing this frozen state and mark the historical fit duration unavailable,
rather than inventing a fit cost.

TRR-0007 provides a bounded public-base fixture using the same K=256 semantics
with its documented CPU-E port: 32 records per domain, one warmup and one
measured call per record, candidate budget 256, proposal budget 512, and
16,320 public-prefix calls per domain. Finance measured and warmup sums were
26.946964 and 27.019813 seconds; Pile sums were 27.274759 and 27.562445
seconds. Model preparation was 2.517929 seconds, peak CUDA allocated/reserved
was 2.399/3.714 GB, and process RSS was 4.179 GB. Its 64-source planning
forecast was 131.789210 seconds for warmup plus measurement. The forecast is a
bounded fixture estimate, not a claim about a TRR-0010 native run.

For a 256-Finance/128-Pile panel, a native TRR-0005-like two-target matrix
would scale to roughly 1,584 seconds of A1+A2 timed work (about 26.4 minutes),
plus four model loads and serialization. A public-base-only 256-Finance/128-Pile
anchor would scale to roughly 792 seconds (about 13.2 minutes). The TRR-0007
CPU-E fixture gives the same order of magnitude. These are planning estimates
from fixed published cells; the final method freeze must record the actual
record batch, target conditions, and prospective resource limit before any
run.

A historical reproduction command shape is available in
`scripts/trr0004_predict_confirmation.py` (the parser binds the public model,
reference helper, lens, panel, registration, and one method):

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:src:scripts \
python3 scripts/trr0004_predict_confirmation.py \
  --repository-root . \
  --panel <one-frozen-TRR0010-panel.json> \
  --selection-plan <frozen-TRR0010-selection-plan.json> \
  --registration <frozen-TRR0010-registration.json> \
  --method frozen_a1_a2_k256 \
  --output-root <create-only-TRR0010-A1A2-output> \
  --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --reference <published-round001-teacher.py> \
  --lens experiments/TRR-0004/evidence/comparators/public_a1_lens.pt \
  --device cuda
```

The placeholders are intentional: TRR-0010 must bind its own single final
panel and registration before execution. The command is not authorization to
reuse a TRR-0004 panel or to open truth. The output must contain the composite
method's prediction and cost receipt before any truth preparation.

## Public capacity and exclusion result

The first aggregate scan is retained as a **pre-TRR9 exclusion diagnostic**,
not as TRR-0010 capacity. It scanned the fixed Pile `[7000,10000)` and Finance
`[12000,20000)` ranges with P08-r2, but did not apply the 384 current TRR-0009
selection-v2 identities and did not union P08-r1 with P08-r2. Its preserved
aggregate output is:

- `experiments/TRR-0010/planning/count_scan/inventory_pre_trr9_p08r2_diagnostic.json`;
- bytes/SHA-256: 25,129 /
  `af41375bb613c9dc531c77751b8d90c21e638d55848a23a914a8b28072c60c4e`;
- Pile 364 and Finance 4,124 were **pre-TRR9 diagnostic counts only** and
  must not be called TRR-0010 eligible capacity.

The authoritative corrected scan reapplied the published TRR7/TRR8
identity ledgers, the union of approved P08-r1 and P08-r2 hash-only ledgers,
and both TRR-0009 selection-v2 identity metadata files. The union contains 514
unique public-record hashes and 514 unique H128 sequence hashes, with 510
r1/r2 overlaps in each field. The current TRR-0009 selection-v2 binding has
256 Finance and 128 Pile rows; only its aggregate counts and file hash were
retained here. No row identity, source text, token ID, selection, model, or
truth was written.

The corrected aggregate output is:

- `experiments/TRR-0010/planning/count_scan/inventory_7000_trr9_p08_union.json`;
- bytes/SHA-256: 25,002 /
  `d64eae6b5e0137f1fb76e89ba7b618455853977330f13058877c6a7c3ab9038b`;
- scanner commit: `4602f98eeb03ac121d8bbc230d7e2b219551b914`;
- elapsed: 10.83 seconds; maximum RSS: 1,209,684 KiB; model and network not
  used; source rows and H128 token IDs were transiently read/derived only for
  identity checks.

| domain and public range | scanned | valid | invalid | remaining unique after all bound exclusions | requested | surplus | remaining commitment SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pile `NeelNanda/pile-10k` `[7000,10000)` | 3,000 | 297 | 366 | 236 | 128 | 108 | `b6c245ae89145e8546e52580a560b35474b7e42524e2a2af6f9b916c09a4bffe` |
| Finance `Josephgflowers/Finance-Instruct-500k` `[12000,20000)` | 8,000 | 4,926 | 2 | 3,867 | 256 | 3,611 | `b16b510db324c9b66be501870e5fdcd2f156871edf6071c855da5997f0afa5d9` |

These are aggregate remaining-capacity counts after all currently bound
identity exclusions; they do not select or reserve a TRR-0010 panel. The
corrected counts support a 128-record-per-domain panel by count. They do not
support a 512-record Pile panel in the fixed `[7000,10000)` range. The
combined exclusion receipt, including input bindings, execution flags, and
preserved failures, is
`experiments/TRR-0010/planning/count_scan/combined_exclusion_receipt.json`
(bytes 16,198, SHA-256
`02a8d5d20b7c3b403c4a53addc2529941aa9058036a25aa4584c1a27e2b3d9a8`). The
P08 union copy used by the scanner is
`experiments/TRR-0010/planning/count_scan/p08_union_sanitized.json` (80,405
bytes, SHA-256
`b6845ab29a65dc07689f855c35c3fedcd5aa8b3c545f0b80098017c126b3ac41`).

A bounded feasibility probe for Pile `[6000,10000)` was attempted with the
same exclusions through a task-local range adapter. It failed closed at the
published partition guard on row 6022 because the repository declares the
Pile holdout as `[7000,10000)`. Extending below 7000 would require a separate
partition/configuration authorization; no fit-row bypass was attempted. The
failure timing receipt is retained in the combined exclusion receipt, and the
6000 extension remains unresolved rather than being counted as capacity.

For final panel planning, a 128-record-per-domain common panel is the
currently supported bounded option. The prior two-target fixture implies about
18 minutes of paired matrix work at that size; root should freeze a
prospective overall limit of 1,800 seconds and per-cell limit of 600 seconds
before any source selection or truth access. A larger Pile panel may be
reconsidered only after a separately approved public partition/range binding;
this count note does not make that choice.

No TRR-0010 source selection or truth access was performed for this note.

## Required TRR-0010 integration boundary

The A1+A2 output must be run on the same final source panel as the other
reported contenders, with the comparator's public-prefix inputs made available
under its own registered public asset bindings. The panel and all method states
must be frozen before one common truth opening. Do not form a post-hoc
intersection of separately selected A1 and A2 panels. The final panel should be
chosen once, then every method, including this single composite comparator,
consumes the declared paired records.

The A1+A2 comparator is reported separately from the current-H full-vocabulary
methods' deployment timing because its candidate simulations and public-prefix
calls are a different inference path. If it is included in a common timing
receipt, use the native boundary above, retain candidate simulations and public
prefix calls, and qualify numerical equivalence against the pinned native
fixture before timing. Do not silently substitute the TRR-0007 CPU-E port for
the native method; label that path as a benchmark-compatible numerical port.

The final handoff should bind, before truth access:

- the composite `frozen_a1_a2_k256` method ID and policy hash;
- the public lens, normalized embedding, public prefix checkpoint/configuration,
  model revision, and all source-code hashes above;
- the TRR-0010 source-selection hash and per-domain count/commitment;
- the exact public-base/public-target conditions actually run;
- prediction artifact hashes, candidate simulation counts, public-prefix calls,
  warmup/measured output equality, synchronized timing, and peak memory; and
- the TRR-0010 unchanged baseline at selected fixed step 400, not the older
  TRR-0007 residual checkpoint.

No inference or truth access was performed for this note.

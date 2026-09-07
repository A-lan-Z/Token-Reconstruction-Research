# TRR-0010 shared-contract review from P09/A2

**Review status:** `CONDITIONAL_RECOMMENDATION_NOT_A_FREEZE`
**Reviewed commit:** `871ae84478b4957919774faa10bfcf126b06d6d9`
**JSON:** `experiments/TRR-0010/planning/shared_contract.proposal.json`, SHA-256 `3b0a68a4a05304e285c1fedcf8a2982bf4bef831e2e5bc0cebb20b8b2dba08b3`
**Markdown:** `experiments/TRR-0010/planning/shared_contract.proposal.md`, SHA-256 `49794625a5264ed8e9ee85140687b219ba8bb3d71c1c61665ce444da536bf66d`

This is a read-only scientific review. I opened only the declared contract and published public metadata references. I did not edit the TRR-0010 worktree, select rows, capture activations, fit a model, open truth, or change the P09 status.

## Decision

The v4 contract is scientifically compatible with the user brief and the P09 proportional-bank recommendation, subject to the binding steps below. Its crossed design, 12,000-row proportional bank, public development role, 128-record-per-domain paired final panel, full-width fit/width-128 evaluation geometry, and source-record uncertainty procedure are all aligned. The contract must remain a proposal until the exact B0/B1 identity ledger, public-development exclusions, deterministic recipe, merged-readout qualification, storage caps, and final panel receipt are signed.

The contract's TRR-0009 `continued_fixed_readout` state (`5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`, step 400) is the agreed TRR-0010 start. The older `2a44...` identifier is retained only as historical inventory metadata and must not appear in any fit, baseline, cost, or gap-closure calculation.

## Exact stage-1 bank recipe for the A2 handoff

The following recipe reuses the published TRR-0007 policy and makes the proportional proposal executable. It is a contract amendment proposal, not an executed selection.

### Frozen inputs and source ranges

Bind these inputs before any source-row selection or public-model forward:

- Public tokenizer snapshot `9213176726f574b556790deb65791e0c5aa438b6`, BOS `128000`, the historical renderer caps (user 1200 characters, output 1200 characters, complete width 192), and public readout/embedding `E` SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`.
- Alpaca: `tatsu-lab/alpaca`, train revision `dce01c9b08f87459cf36a430d809084718273017`, pinned Arrow SHA-256 `f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794`, all rows eligible only after the exact exclusion ledger.
- Pile: `NeelNanda/pile-10k`, train revision `127bfedcd5047750df5ccf3a12979a47bfa0bafa`, published Arrow SHA-256 `77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113`; additions scan only `[2000,7000)`.
- Finance: `Josephgflowers/Finance-Instruct-500k`, train revision `583a98fb0ec14d904e9423b671d9d0fea88891b6`, published shard SHAs `b49ca0980a0b02fecbef2220eee0ef5d3c3c893ae42b4e1910edec993c3d164e` and `ce4b0786646cd68561da736f145fd5df7ba2f4e754e0caa3ae646d6be9900bd3`; additions scan only `[2000,12000)`.
- Preserve the exact published 1,200-record B0 payload and identity/activation manifest as a byte-equivalent prefix. The B0 manifest, token/mask/position artifacts, H payload, record order, and all tensor digests remain pending A2 binding.
- Reserve the Agent-1 natural final-panel ranges Pile `[7000,10000)` and Finance `[12000,20000)` exclusively for the later 128-per-domain panel. The failed Pile `[6000,10000)` extension remains excluded. The aggregate eligible counts 236 Pile and 3,867 Finance are availability checks, not selection.

The complete namespace-specific exclusion union must be applied before eligibility: B0/current fit IDs, TRR-0004 fit and public-development IDs, TRR-0007 current-bank IDs, P04/P08 approved opaque reservations, the opened TRR-0009/Agent-1 public-development identities, and any later root-approved validation/final-panel identities. A rendered-text SHA-256, a sequence digest, and a reservation digest are different namespaces unless the receipt states their exact canonical bytes; the P09 audit's 4,193-value union and zero rendered-text hits are an upper bound diagnostic, not proof of disjointness.

### Five source strata and exact post-BOS length quotas

Use the published B0 `length_vector_digest` `b8b3392d6984e7e109ad70108cb36aaa861555eccce4b4bf14e3aea5c2846bb8` and its four inherited post-BOS bins: `40–63`, `64–95`, `96–127`, `128–191`. The exact B1 totals and additions are:

| source stratum | B0 rows | B1 total | B1 addition | B1 bin totals `[40–63, 64–95, 96–127, 128–191]` | addition bin quotas |
| --- | ---: | ---: | ---: | --- | --- |
| Alpaca natural | 600 | 6,000 | 5,400 | `[1390, 2260, 1610, 740]` | `[1251, 2034, 1449, 666]` |
| Pile natural | 300 | 3,000 | 2,700 | `[620, 890, 570, 920]` | `[558, 801, 513, 828]` |
| Finance natural | 180 | 1,800 | 1,620 | `[360, 570, 300, 570]` | `[324, 513, 270, 513]` |
| Pile controlled | 60 | 600 | 540 | `[0, 0, 0, 600]` | `[0, 0, 0, 540]` |
| Finance controlled | 60 | 600 | 540 | `[0, 0, 0, 600]` | `[0, 0, 0, 540]` |
| **total** | **1,200** | **12,000** | **10,800** | **`[2370, 3720, 2480, 3430]`** | **`[2133, 3348, 2232, 3087]`** |

Within each source/bin, inherit the published exact-length histogram rather than reshaping it after seeing capacity. The target count for each exact post-BOS length is ten times the B0 count, with the B0 prefix contributing one copy and the additions contributing nine copies. Require full rendered-token length at least target length plus BOS, clip only after BOS to the exact target length, and fail closed if any exact length quota cannot be filled. Iterate the inherited target-length slots in their published descending-length/slot order and record that allocation order in the receipt. This preserves the proposal's proportional mixture and makes the 10x natural-support claim auditable.

### Stable ranking and public-development exclusions

For each eligible natural source row, sort by the published stable key

```text
SHA256("TRR-0007|7007|{dataset_id}|{split}|{revision}|row:{row_index}")
```

ascending, with the complete dataset ID, split, revision, and integer row index bound in the receipt. For every source/bin/length quota, consume the first remaining row in that order after applying all keyed identities and namespace-correct digests. Do not sort by model scores, error labels, final answers, or token support discovered from evaluation.

The 384 opened Agent-1 `public_base` rows are public development only. Their exact ordered record/sequence identities must be added to the exclusion ledger **before** B1 additions are selected. The later final panel is selected once from the reserved ranges after all methods and exclusion ledgers are frozen; it is not a new validation pool and cannot influence bank selection. Validation and final rows, their derived sequences, B0/B1 fit rows, and known opened-study rows must be represented by exact record IDs, parent IDs, source-row keys, rendered hashes, and namespace-labeled sequence hashes.

### Controlled replacement and support-token policy

Controlled parent rows must have post-BOS length 128–191. For each controlled record, retain the inherited four one-based post-BOS position bins and quotas:

- positions `1–15`: 3 replacements;
- positions `16–39`: 6 replacements;
- positions `40–79`: 9 replacements;
- positions `80–127`: 12 replacements.

That is 30 occurrences per record, with structural BOS/PAD positions removed. Choose each bin's ordered positions by the published digest rotation and midpoint rule, store zero-based offsets, run one registered public cut-4 forward over every complete constructed sequence, and reject activation splicing. At B1 this produces 18,000 controlled replacement occurrences per source (36,000 total), with the 120 B0 controlled records as the unchanged prefix.

For replacement token identities, reuse the exact published ordered 3,600-ID support list and artifact, including its order and combined hash `0e84f625e24fb4a2038972cf417923dc20a8cc2fd85e55ac6e9b7a846f5ca576`. Do not re-select a new 1,600-ID suffix or infer support from a later enriched fit map; the earlier baseline/addition distinction is historical provenance only. Bind the exact published artifact and hash, and do not mix the recipe's v1 and bank-v5/v2 frequency-file aliases. Assign the frozen 3,600-ID list in ten deterministic cycles over the 36,000 B1 controlled occurrences, preserving the B0 first cycle; record the resulting occurrence/order digest. Natural-row token support is whatever the frozen public tokenization produces, and the directional supported-row set is derived from fitting data only, never evaluation answers.

### Fixed diagnostic before fit

Freeze one score-independent 64-record diagnostic for each bank before fitting, shared by that bank's four crossed arms and reused through the nested B0 prefix:

- Alpaca natural 32, Pile natural 16, Finance natural 10, Pile controlled 3, Finance controlled 3;
- use frozen seed `4010` and, within each stratum, rank the bound record identities by `SHA256("TRR-0010|fixed-diagnostic|4010|{record_id}")` ascending;
- bind the ordered record IDs, source labels, exact sequence/position masks, and digest before the first schedule read or optimizer step;
- use it only for fixed diagnostic curves/health checks, never checkpoint selection, source selection, or final evaluation.

The diagnostic has identical identities and predeclared positions across current/expanded banks and fixed/directional readouts. Any missing B0 stratum or changed prefix is a binding failure, not a reason to substitute rows after training begins.

### Cross-stage identity and geometry bindings

The receipt must distinguish the complete width-192 fit objects from the width-128 evaluation objects. In particular, the B8 complete-192 sequence set must be distinct from the 512 non-BOS valid-loss draws; capture size is a separately qualified resource quantity rather than an inference from fit memory. Bind the exact mapping from each width-192 fit record/hash to its width-128 reservation and parent ID. Sequence hashes computed at mismatched geometries are not comparable and must never satisfy an identity check. The B0 prefix and the 64-record diagnostic binding must be written and verified before any schedule is read.

## Alignment findings

The following contract choices are approved in principle:

1. The four crossed arms and separate unchanged starting reference satisfy the comparison brief; no A2 deployment, contextual input, teacher loss, staging schedule, or ensemble is introduced.
2. The 12,000-row proportional B1 totals exactly match the P09 recommendation and preserve the 10% controlled share. Parent sources, derived variants, valid positions, support IDs, repeated exposures, and optimizer updates are reported separately.
3. Full width-192 fit geometry, width-128 natural evaluation clips, actual public-model forwards, nested B0 prefixing, equal optimizer opportunities, and a continued-training fixed control are appropriate.
4. The 384-record public-base development panel and the 128 Finance + 128 Pile two-target final/A1+A2 panel keep roles distinct and preserve paired changed-target analysis. Domains remain separate.
5. The 10,000-draw source-record bootstrap, paired conditions, separate domains, exact-recovery metric, gains/regressions, and predeclared useful/harm gates are compatible with the charter and the no-tiny-gain instruction.

## Required amendments before stage-1 sign-off

1. Bind the exact B0 payload/record-order/activation hashes, public model `E` hash, tokenizer revision, public source revisions, the five quotas above, exact length histogram digest, frozen ordered 3,600-ID support artifact, stable ordering formula, and the seed-4010 diagnostic in one create-only selection receipt. Include the width-192-fit to width-128-reservation/parent mapping and keep B8 complete-192 sequences distinct from the 512 non-BOS valid-loss draws.
2. Add the 384 public-development identities to the exclusion ledger before B1 selection. Verify zero overlap separately for B0, B1 additions, validation, the reserved final panel, and all approved P08/TRR-0009/opened-study ledgers using namespace-correct canonicalization.
3. After actual B1 valid positions are bound, evaluate the contract's existing formula for the exact common `N` and report per-arm draws and repeated exposures; the current-bank control will have a different repeat rate by design. The expected roughly 1.24M positions make 12,000 a lower-bound grid point, but the receipt must determine the actual value.
4. Keep the resource caps explicitly pending setup qualification. The 12,000-row BF16 H payload alone is `12,000 × 192 × 2,048 × 2 = 9,437,184,000` bytes (8.79 GiB), before token/mask/position sidecars and staging overhead; the full directional merged-readout estimate of 11.477 GiB against a 12.2 GiB live-free snapshot is not a qualification margin. Measure and bind capture caps before capture. After the B1 support and capture objects are prepared, qualify the largest expanded-directional merged-W cell and bind measured disk, host-RSS, isolated GPU peak, wall-time, and output-equivalence caps before releasing the full matrix.
5. Bind A1+A2 state/input/timing identities before using the 0.25 candidate/A1+A2 runtime gate; until then that comparison remains UNKNOWN. Do not use the 262–267 second planning timing as a scientific result or as a substitute for a receipt.
6. Clarify in the frozen contract that the TRR-0009 `5cada...` checkpoint is the only TRR-0010 starting state and that P09's older `2a44...` state is not mixed into any arm, baseline, cost, or gap-closure calculation.

## Disposition

`CONDITIONAL_RECOMMENDATION_NOT_A_FREEZE`: approve the crossed study direction and proportional bank recommendation for root review. Hold stage-1 public bank selection/capture until the exact identity/exclusion/recipe receipt and pre-capture resource caps are jointly signed; release the full matrix only after the prepared-support largest-cell qualification. No scientific conclusion or promotion decision is made by this review.

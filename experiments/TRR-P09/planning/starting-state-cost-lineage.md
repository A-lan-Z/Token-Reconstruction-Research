# TRR-P09 starting-state cost lineage

**Audit date:** 2026-09-07  
**Scope:** the public, published lineage needed to produce the P09 starting
checkpoint `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`.
This is a cost ledger, not an additive grand total.  A value marked
`UNKNOWN` was not measured and is not treated as zero.

The direct learned-state chain is:

```text
public Llama snapshot
  -> TRR-0005 public activation/corpus preparation
  -> TRR-0007 current_enriched residual fit, selected step 2100
       state 2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8
  -> TRR-0009 continued_fixed_readout fit, selected step 400
       P09 state 5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14
```

TRR-0009 also fitted an improved public bank and ran an adaptable-readout
control.  Those runs are included below as support or comparison work where
they affected the published continuation, but neither is silently described
as a weight ancestor of `5cada…`.

## Identity and accounting boundary

The published TRR-0009 manifest is from commit
`4602f98eeb03ac121d8bbc230d7e2b219551b914` and has SHA-256
`00d2e9bcb75962072dabdac50ad38b32e9ad638fae77c9da75faec1760048e64`.
Its `starting_state` records the TRR-0007 current-bank residual at selected
step 2100, 29,390,628 bytes, SHA-256
`2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8`.
The P09 checkpoint is the TRR-0009 fixed-continuation state at selected step
400, 29,390,492 bytes, SHA-256
`5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`.

The TRR-0009 manifest explicitly defines the accounting boundary: shared
preparation is charged once; arm walls, phase walls, and whole-run wall are
reported separately; inherited TRR-0005/TRR-0007 measurements are not added to
the incremental TRR-0009 continuation cost.  The ledger follows that rule.

## Full-run cost ledger

All fit schedules below ran 3,000 optimizer steps unless stated otherwise.
“Selected step” is the retained checkpoint; it does not truncate the cost of
the preceding full run.

| Phase and role | Full run and selection | Wall / updates | Preparation, capture, and selection fields | Evidence and accounting note |
|---|---|---:|---|---|
| Public base model and normalized embedding table | The public Llama snapshot is an input, not a locally measured pretraining run | **Pretraining wall, optimizer updates, acquisition cost: `UNKNOWN`** | Normalized embedding table is 1,050,673,488 bytes (SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`); load time is a runtime measurement, not pretraining cost | Model revision `9213176726f574b556790deb65791e0c5aa438b6`; no base pretraining receipt is present in the published lineage |
| TRR-0005 corpus and public activation used by later public fits | Corpus preparation complete; no decoder state from its fits initializes TRR7 | Full corpus preparation **20.013262748994748 s**; updates not applicable | Public activation capture **11.447659793004277 s** authoritative child interval; full launcher interval **17.839759826660156 s** | `experiments/TRR-0005/corpus_run/run_receipt.json`, SHA `a68c5b61b66e499d24db4696726db75bce7972602a3ed0c25f1baec99e419abb`; `public_activation_v1/capture_manifest_receipt.json`, SHA `48c469373de931b905af642a21977b2fe85877208db16ec9cc13adc88fec5a7a`. Shared activation/corpus support, not a decoder-state initializer |
| TRR-0005 historical decoder-fit campaign | Eight complete public fits: two 3,000-step run groups (6-arm joint fit plus 2-arm qknorm fit), full run elapsed **588.7935206120019 + 212.90698062499723 = 801.7005012369991 s** | **8 × 3,000-step arms**; individual selected steps vary by arm | Selection used earliest maximum public validation style-balanced accuracy including step 0; standalone selection wall is **UNKNOWN** | `joint_fit_v1/run_evidence.json`, SHA `87c1901684853ab43c7ce3d5660999c02fa38744aefef89cd2ee44d89eafac0a`; `joint_fit_qknorm_v1/run_evidence.json`, SHA `95aefffa1abc61cbf8dbc102c084ab152d4bd7a5f66658735b4b4b5021b68a3a`; historical fit context only. TRR7 is initialized from neutral/identity-affine parameters, so these decoder fits are not weight ancestors |
| TRR-0007 current-bank fit producing the direct TRR9 prefit state | Current-enriched bank run; residual arm selected step **2100 / 3000**; sibling current positionwise arm selected step **1600 / 3000** | Whole run **262.50082792500325 s**; residual arm internal fit wall **123.13956650999899 s**; residual train **90.7981688100408 s**, validation **3.6432379160105484 s**, challenge **22.970867949945386 s** | Accepted public-bank preparation in TRR7 manifest **23.666454504 s**; standalone current-residual selection wall **UNKNOWN**. The fit wall and validation component already contain the work used to select step 2100 | `experiments/TRR-0007/enriched_fit_v1/run_evidence.json`, SHA `80959b3bac25d76158a0fcaf9381a71915c3d5bc19f4f63faa15665f721d5cc6`; `current_enriched/bank_result.json`, SHA `4900201c3bcb3e446a71d851f5ab67c337d2b87fed982527f2b30a31e6d60a13`; TRR7 manifest SHA `e0e0bca00746825edffde0960e793a3877d74098a3d901e0c327925449ea4a7b`. This is the direct state ancestor |
| TRR-0007 improved public bank support fit | Separate improved-bank run, not the P09 state ancestor; both methods ran the same 3,000-step schedule and the improved residual fit selected step 2100 | Whole run **263.88357899199764 s**; improved residual bank-internal fit wall **123.50211520800076 s**; improved-bank run internal wall **263.5049643430102 s** | Improved activation capture **8.22466169699328 s**; bank-construction receipt execution **11.330641836000723 s**. These are separate receipt scopes; the accepted TRR7 public-bank preparation charge above is not added to either here | `experiments/TRR-0007/improved_fit_v1/run_evidence.json`, SHA `994f791d34a701d4064bff43a33250bc47cccc94ed475f5f24845bd37b1bbb61`; `improved_fit_v1/improved_public_bank/bank_result.json`, SHA `c103166b53a267fdb01e1876ce73af90769cddcc5038a0e110d8120b3eb21b6c`; capture receipt SHA `5a892ba16d7dc0a2f22668f4672498ee9a51fbb0f78a78f5ef5cf74a8ebe0aa5`. TRR9 used the published improved bank for continuation fitting/validation, so this is support/selection context, not a hidden weight ancestor |
| TRR-0009 continuation fixed arm producing P09 | Full fixed arm **3,000 steps**, retained selected step **400** | Arm wall **355.82272331698914 s**; gradient-update component **92.3284733351611 s**; evaluation/validation component **260.987909478019 s** | Shared continuation preparation **9.148822391987778 s, charged once**; challenge **3.878089745005127 s**; standalone selected-step/selection wall **UNKNOWN** | `experiments/TRR-0009/training/run_v1/continued_fixed_readout/receipt.json`, SHA `96db430897d79e22cbc1e57c5736b2a1b370fdb76cf48559b34b4bf774d88218`; fixed state is P09 `5cada…` |
| TRR-0009 adaptable control arm | Full adaptable arm **3,000 steps**, retained selected step **400** | Arm wall **393.80261707899626 s**; gradient-update component **107.9819799256511 s**; evaluation/validation component **283.77045111803454 s** | Same shared preparation receipt **9.148822391987778 s**; standalone selection wall **UNKNOWN** | `continued_adaptable_readout/receipt.json`, SHA `e47b90f45e2716710b28dccaa1c6282cc5300e7f4fc11485c94ed5dacbfaa08f`; sibling comparison, not the P09 state |
| TRR-0009 unchanged anchor | No optimizer updates; selected step 0 | Anchor wall **8.388725268974667 s**; gradient updates **0** | Do not charge the repeated preparation field again; challenge field is part of the run receipt and is not added to the whole-run wall | `unchanged_anchor/receipt.json`, SHA `11b0acc5ddc810010280ff3179dafe1f3220cba1c8715788f2bc2455db5ff3ca`; matrix control, not an additional trained ancestor |
| TRR-0009 continuation run boundary | One process/run containing the continuation arms and anchor | Whole-run wall **770.0699744224548 s**; arm-wall sum **758.0140656649601 s**; difference **12.055908757494763 s** | Shared preparation **9.148822391987778 s once**; challenge **3.878089745005127 s**; no arithmetic sum of lifecycle components | `experiments/TRR-0009/training/run_v1/run_receipt.json`, SHA `6ad636b789fc6b39b432a1ce67c1f2347d627f7655fafa55244404d295da18d2`. The whole-run wall is authoritative for the run boundary; arm rows explain allocation |

## Historical reference work, not direct weight ancestry

TRR-0004 supplied a retained affine comparator used for diagnostics in later
work. It did not initialize the TRR7 residual: TRR5 and TRR7 designs specify
neutral/identity-affine initialization. Its published large affine run had two
3,000-step methods with fit walls **94.42281889100195 s** and
**97.2540266520009 s**, launcher wall **198.69985462199838 s**, selected step
1900; the small diagnostic run had fit walls **93.03152749899891 s** and
**96.76509686799909 s**, launcher wall **196.89417636099824 s**, selected step
150. Successful public activation v2 preparation was **9.764313294999738 s**.
These are reference/diagnostic costs and are not added to the direct P09
lineage. The TRR4 manifest is
`experiments/TRR-0004/manifest.json`, SHA
`60a4c427e18733989f44ac51febff8d6afb37c23f4848020d53b63da1aa0d4d8`.

The historical A1 fit and its acquisition/preparation provenance are
explicitly **UNKNOWN**. No A1 cost is inferred from a retained artifact or
from the TRR4 launcher wall.

## Selection and double-counting rules

* Count every complete scheduled fit that was part of the published choice or
  continuation: a selected step of 400 or 2100 does not turn a 3,000-step run
  into a 400- or 2,100-step cost.
* A validation component is reported where the receipt exposes it. A separate
  selection wall is `UNKNOWN` when the receipt does not isolate it; it is not
  inferred by subtraction.
* The TRR-0009 whole-run wall and its arm-wall sum are alternative boundaries,
  not addends. The fixed, adaptable, and anchor rows explain the matrix but
  must not be summed with the whole-run value.
* Shared preparation is charged once within its run. The repeated
  `preparation_seconds` field on each TRR9 arm is not three preparations.
* The TRR7 accepted public-bank preparation value, bank-construction receipt,
  and improved activation capture have different receipt scopes. They remain
  separate rows until a published accounting boundary says otherwise.
* TRR-0005 and TRR-0004 fit campaigns are retained as historical context and
  are not weight ancestors of `5cada…`; they are not silently folded into a
  total. TRR7 improved-bank fitting is similarly shown as continuation support
  rather than a direct state ancestor.
* Future P09 source-bank selection, capture, and fitting costs were not run in
  this audit and are `UNKNOWN`/not applicable to this historical starting-state
  ledger.

There is therefore no defensible single grand-total seconds value in this
record. The base model's pretraining/acquisition cost, historical A1 cost,
standalone state-selection times, and any receipt scope not explicitly
resolved above remain `UNKNOWN`.

## Access boundary

This audit read published JSON/Markdown metadata and receipt hashes only. It
did not open private or evaluator truth, model/payload tensors, or hidden
holdouts, and it did not execute a fit, capture, or selection.

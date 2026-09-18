# Frozen fragment mechanism and independent R2 confirmation

## Selection and method
On opened R1, prefix embedding64+intrinsic MLP64 plus deterministic tokenizer suffix expansion atK512 achieved complete candidate recall in matched/shifted natural2032/2032 and stress312/312. K256 missed12 and8 natural tokens; the embedding-only expansion missed natural and stress. Select exactly the union+suffixK512 policy before constructing the new panel. This is candidate evidence only.

The intrinsic map uses the same prefix's embeddings, RMSNorm and MLP weights, with attention omitted for this disposable lookup calculation. Geometry comes from that prefix's attention output and MLP down weights. Every derived table is invalidated after prefix mutation/replacement. Token suffix metadata comes only from the already-required public tokenizer. No independently trained parameters, auxiliary prediction checkpoint, optimizer, examples, or gradients. Full-vector lookup is followed by ordinary native current-prefix A2 verification and reconstructed-history commitment. Keep512 candidate slots including duplicate padding; do not silently claim deduplication savings.

Native intrinsic build is256 rows. A4096-row build may be used ONLY if the whole-vocabulary BF16 intrinsic outputs are bitwise equal to256 during qualification; otherwise preserve its discrepancy and exclude the faster build. Dictionary transformation remains256-row FP32 to preserve the development candidate identity.

## Qualification and resources
Qualify all three methods at max128positions andK512 before fresh capture. Compare complete candidate identity to the old development fragment probe on one already-opened record. Match original A1+A2 candidates and tokens on the independent synthetic fixture. Tokenizer batched cache must equal direct per-token suffix implementation on106 fixed public token IDs. Unit tests cover decomposition, ordering/budget, and prefix invalidation.

Worst geometry: public prefix~1.01GB; two FP32 proposal tables2.10GB; baseline embedding table1.05GB;512-candidate KV expansion at127 positions~0.27GB plus copy/sublayer workspace; table-building temporaryBF16~0.53GB is freed before decoding. Expected reserved<6GB with live16GB GPU, prior free~11GB. Watchdog requires>=2GiB freeGPU,>=8GiB host, temp<80C, RSS<10GiB. All operations no-grad. Qualifier180s, prediction900s; qualify maximum cell and retain measured margin before matrix.

## Independent panel
Seed141410. Sample16 qualifying paragraphs per Gutenberg11 and84 (32 natural clips128positions includingBOS), excluding all prior Agent4/rescue sources and all R1 paragraph hashes. Generate8 new40-position identifier clips with the new seed. Pair identical40 sources across matched public target and actual TRR-P12 rank8 q/v stage256 prefix-LoRA target.80 observations total. Same revisions, masks, native BF16/FP32RoPE capture and evaluator-only target adapter asR1. No new target training. New source freshness is task-local only.

## Complete scoped matrix
Three frozen arms on all80 observations: fragment512, identical embedding/MLP union128 without suffixes, and original fitted public A1+A2 directcosK256.240 unique cells. No current truth before all240 outputs freeze. Three identical synchronized repetitions percell, rotate method order. Time observation transfer/proposal/native verification/cachecommit/outputtransfer equally; report serialization separately. Include cold tokenizer metadata preparation, first/repeat prefix-table rebuild, and prefix loading; report both amortized and one-rebuild-per-record cost. No historical A1 training cost is invented.

Scorer validates complete matrix, code/input/output hashes, shapes/ranges, actual argmax decision, finite scores, and three timing repetitions before labels. Missing-cell/corrupted-output/changed-code tests must fail. Report natural and stress separately for each condition; paired record bootstrap within each. This is an exploratory promising-direction test, not canonical comparison completeness, recovered-prefix integration, cold-start tracking, broad population performance, or a replacement declaration. Canonical active registry unchanged.

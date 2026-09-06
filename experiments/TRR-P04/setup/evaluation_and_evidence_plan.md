# TRR-P04 setup and evidence plan

Status: proposed and unfrozen. This file records the setup contract after a read-only review of the immutable TRR-0004/PR7 snapshot. It contains no selected evaluation IDs, source rows, token IDs, teacher scores, target-update weights, or evaluation truth.

## Parent and reusable public assets

The worktree is `/tmp/trr-p04`, branch `task/TRR-P04`, rooted at `6e8b683e404c0acb70cd59b7dd6d6868b2061f61` from open PR #7 (`task/TRR-0004`). The pinned model/tokenizer is `meta-llama/Llama-3.2-1B-Instruct`, revision `9213176726f574b556790deb65791e0c5aa438b6`, with BOS `128000`, padding `128001`, cut depth 4, and hidden width 2048.

The PR7 metadata identifies the public Pile source as `NeelNanda/pile-10k` revision `127bfedcd5047750df5ccf3a12979a47bfa0bafa`, the Finance source as `Josephgflowers/Finance-Instruct-500k` fingerprint `4abbac8acaab4205`, and the Alpaca source as `tatsu-lab/alpaca` revision `dce01c9b08f87459cf36a430d809084718273017`. Their pinned arrow hashes are recorded in the parent manifests and are reused only as read-only provenance references. The PR7 public normalized embedding table is `outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors`, SHA-256 `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`; the historical `public_lora_2601` target is retained as a prior diagnostic identity and is not an unseen P04 target.

No model or large tensor was loaded during this setup review. No P03 asset, holdout, mutable worktree, global state, active registry, or evaluator answer was opened.

## Fresh evaluation and source separation

Use 72 independent natural source records arranged as three styles, four post-BOS length strata (`16, 32, 64, 128`), and six records per style/length cell. The proposed styles are plain Pile, Finance chat-template rendering, and Alpaca instruction rendering. The exact selection seed and evaluator-target-update seed must be fixed by root/implementation before selection, then written to the frozen panel metadata. Selection is by deterministic source-row order after excluding every exact PR7 fit/validation/panel identity and any duplicate normalized source-text or truncated-token hash. A whole source record is excluded on collision. Source text and token IDs stay out of attack metadata and publication.

Both target conditions use the identical source-record clusters: `public_base` and a newly seeded `p04_evaluator_target_update_v1`. The latter is evaluator-side only and is not available to student fitting, teacher evidence generation, or panel selection. Its lineage, seed, update data identity, and activation drift against `public_base` are recorded separately. PR7's `public_lora_2601` is not treated as this fresh condition because it was already used as a public synthetic diagnostic.

Predeclare a 12-record native A1+A2 anchor: four records per style in the 32-token stratum, yielding 384 post-BOS positions per target. Keep the anchor denominator separate from the 72-record panel. If the available implementation is a numerical port, retain the exact normalization and denominator label and do not substitute it for a native claim.

## Public training and teacher qualification

Create three disjoint public pools: the immutable 1,200-record PR7 public fit/replay pool, a 256-record correction pool with public-base mistakes or uncertainty, and a 192-record public validation pool. The correction pool is selected from public development information and is disjoint from fitting, validation, and fresh evaluation records. All three student arms see the same records, labels, row order, sampling schedule, and optimizer opportunity.

Generate teacher evidence only on 384 correction-pool positions: 256 predeclared difficult positions and 128 seeded random-audit positions. Use one frozen privileged public-prefix A2-style scorer with candidate/proposal budget K=32. The separate native A1+A2 evaluation anchor uses K=256 and is never substituted for the training teacher. Candidate identities are generated once from the same frozen public decoder and reused unchanged by H and D. Record proposal misses, candidate recall, teacher correctness, teacher fixes and introduced errors, true-token inclusion, score span/entropy, and finite-precision ties. Evidence made with known public prefixes is privileged training evidence; it cannot be reported as native BOS-only reconstruction.

S is full-vocabulary CE. H adds a fixed label-derived hard-confusion loss over the frozen candidate IDs, excluding the gold token and using a single public-validation choice of weights/margin. D adds one fixed centered relative-score or pairwise-ranking objective from the teacher evidence. Choose its scale once on public validation after checking score range and ties; do not run a loss sweep. All arms, including the `affine_same_data` reference, use the same trainable affine path during fitting; at inference all arms predict unrestricted full-vocabulary outputs from activation prefixes, with a reset unidirectional GRU, competent affine/identity path, tied public embedding output, and no source tokens, candidates, teacher scores, labels, prefix calls, or guessed-token feedback.

Use paired training seeds 1737 and 2711 for every `affine_same_data`, `student_s`, `student_h`, and `student_d` arm. A same-data affine reference is retained so any recurrent gain is separated from simply receiving extra public training. Training, correction, and validation labels remain public training truth and are never mixed with evaluator-private truth.

## Freeze, scoring, and uncertainty

The evaluator stores observations and an immutable panel index without labels. Private evaluation truth uses separate per-target JSONL files with rows `{record_id, token_ids}`. Private evaluator-target weights stay in a separate directory. Freeze the panel selection, overlap audit, observation index, all student/reference weights, teacher evidence, candidate IDs, and prediction roots before any scorer opens evaluation truth. The scorer reads frozen predictions and then writes metrics; it never rewrites predictions or feeds answers back into selection or training.

Report token accuracy, complete records, per-style/per-length/per-target results, and paired source-level gains/regressions. Bootstrap `source_record_id` clusters so the two target conditions do not count as independent observations. Keep teacher agreement/shortlist recall secondary. Record public activation generation, teacher simulations, student fitting, retained state/table bytes, startup, and warmed uncontended inference separately, together with guarded command, timing, RSS, host-memory, and failure receipts.

The publication set should contain only task-sized predictions/diagnostics, compact score and freeze receipts, safe runtime evidence, source/config/code hashes, and the report/manifest. Exclude evaluator truth, evaluator-target weights, source rows, private indexes, large observation tensors, and model/embedding payloads. Any failed or superseded preparation remains local with a reason and hash; it is not silently reused.

The decision is exploratory: attribute a gain to recurrent architecture, hard-example label supervision, or teacher relative-score information. A negative D-versus-H result is a valid outcome; retain H if it is simpler and better. No result from this setup should be framed as an A2 replacement or a canonical benchmark completion.

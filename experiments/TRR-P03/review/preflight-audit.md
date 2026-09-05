# TRR-P03 Stage 1 preflight audit

**Audit date:** 2026-09-06  
**Disposition:** HOLD before opening Stage 1 truth

This is a static and CLI/interface audit. It did not load a model, boundary
table, projected table, observation payload, or truth sidecar, and it did not
run truth scoring. The existing focused model-free checks are recorded in
`experiments/TRR-P03/runtime-status.md` (13 focused tests passed); they are not
repeated here.

## Checks that are ready

- The frozen setup receipt declares 24 Stage 1 records, six each at scored
  lengths 16, 39, 64, and 128, with 1,482 scored tokens per target. The exact
  anchor is `p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`, and
  `p03-s1-r0012`, as recorded in
  `experiments/TRR-P03/setup/panel-20260906-frozen/setup-receipt.json` and
  `experiments/TRR-P03/plan.json`.
- The public I/O loader rejects truth-bearing fields, verifies artifact hashes,
  checks `[records, scored_tokens + 1, 2048]` geometry, requires fully active
  masks, and checks contiguous positions with BOS at position zero. The
  reconstruction flattening path repeats the mask, position, and sequence
  checks before reading public assets.
- The model-bearing CLIs set `CUDA_VISIBLE_DEVICES` empty, Torch intra-op and
  inter-op threads to 8 and 1, fixed seeds, and deterministic algorithms.
  Standalone raw/projected/A1 ranking uses float32 scores with descending
  score and ascending token ID on exact finite-precision ties. Native A1 keeps
  the FP32 `exp(s)`-scaled logits for decisions; cosine-equivalent division is
  reporting only. The anchor separately retains the published `torch.topk`
  proposal order and first-argmax rule.
- `prepare_projected.py` constructs a float32 full table once and records its
  source prototype and lens hashes. This is usable for both arms only when its
  output is passed to every reconstruction invocation; the reconstruction CLI
  otherwise has a per-invocation projection fallback.
- The current watchdog source records only its explicit safe environment
  allowlist and marks the receipt redacted. Freeze this revision before any
  run; the earlier smoke receipts that were sanitized in place are utility
  evidence only and must not be used as the scientific run receipts.

## Blocking findings and minimum actions

1. **No strict Stage 1 joint pre-score validator exists.**
   `verify_prediction_bundles()` accepts one root, makes the three base methods
   mandatory only when one of them is present, and treats the A1+A2 method as
   optional. `score.py` also makes `--paired-prediction-root` optional. Thus a
   syntactically frozen but incomplete bundle can reach `_load_truth()`.

   Add or run a small strict pre-score validator that writes an immutable
   PASS receipt before the real score command. Keep the generic scorer for
   synthetic fixtures, but the Stage 1 path must fail closed unless the
   receipt verifies all of the following:

   - exactly two immutable prediction roots, one for each evaluator target;
   - `truth_opened=false`, matching plan hash and implementation commit, and
     valid hashes for every freeze entry and referenced evidence/config/asset;
   - the exact ordered IDs `p03-s1-r0001` through `p03-s1-r0024`, with
     sequence lengths `[17]*6 + [40]*6 + [65]*6 + [129]*6`;
   - all three static methods over all 24 records in each root;
   - `historical_a1_a2_anchor.cosine` over exactly the four anchor IDs above,
     with no substituted or expanded anchor set;
   - identical method sets, ordered IDs, sequence lengths, observation index
     identities, mask/position digests, and shared source/config/asset hashes
     between the two roots; and
   - a final scoring invocation that always supplies
     `--paired-prediction-root` and does not supply
     `--allow-unequal-strata`.

   The receipt must be created and hash-checked before the first read of
   `stage1/private_truth.jsonl`.

2. **The runtime observation contract and grouped-index writer disagree.**
   `experiments/TRR-P03/setup/interface.json` requires per-record
   `mask_digest` and `position_digest`, but
   `scripts/trr_p03/generate_observations.py` does not emit them in grouped
   descriptors and `io.validate_observation_index()` fills empty strings when
   expanding those descriptors. The loader verifies masks and positions from
   tensor contents, but the current index cannot certify the required
   cross-target identity. Emit the required digests in the runtime index (the
   preferred fix), or have the strict pre-score validator compute and record
   them from both truth-free observation bundles and reject any missing or
   unequal digest. Do not certify the current empty-digest index as complete.

3. **There is no cross-target observation comparison in the current loader or
   scorer.** Each index is validated independently; neither path compares the
   two ordered IDs, sequence geometry, mask bytes, or position bytes. Before
   truth, the strict receipt must load both public indexes and compare those
   fields and their per-record digests. The target-condition map remains
   evaluator-only, but the receipt must bind bundle-a to the pinned matched
   base model and bundle-b to the pinned full-SFT Vikhr snapshot. The current
   generator checks model geometry but does not bind `bundle-id` to either
   model identity, so this mapping needs an evaluator command/manifest check.

4. **Do not use the generator's `--stage all` path.** It groups Stage 1 and
   Stage 2 rows of the same length into one descriptor labelled by the first
   stage. Generate Stage 1 and Stage 2 in separate create-only output roots;
   leave Stage 2 truth sealed until the Stage 1 gate and compact-method
   constants are frozen. Never pass a private truth path to reconstruction.

5. **Resource qualification is still missing.** No real model, full table,
   projected table, or native A1+A2 cell has been run under the current
   source freeze. Wrap preparation, each target observation generation, and
   each reconstruction in `resource_watchdog.py` with the declared defaults:
   `max-rss-bytes=8589934592`, `min-available-bytes=10737418240`, and a declared
   timeout. A clean qualification requires watchdog PASS, child exit 0,
   sampled peak group RSS below 8 GiB, minimum sampled host availability at
   least 10 GiB, readable live resource samples, and no truth-bearing input.

## Minimal largest-cell qualification

After the strict source/config freeze and truth-free target observations are
available, qualify the largest cell first. The qualification is followed by
the complete 24-record matrix, then the strict joint receipt, and only then
the first read of Stage 1 truth.

1. Run `prepare_projected.py` once under the watchdog. Record the projected
   artifact hash and verify its raw-table and lens hashes. Pass that same
   `--projected-prototype` file to both target runs.
2. Derive a create-only, truth-free qualification index for each target with
   the four anchors plus one length-128 record, for example
   `p03-s1-r0007`, `p03-s1-r0009`, `p03-s1-r0011`, `p03-s1-r0012`, and
   `p03-s1-r0019`. Preserve the original opaque IDs, sequence geometry, mask
   and position bytes, and artifact provenance; do not open or copy truth.
   Using the complete 24-record Stage 1 index is an acceptable larger
   fallback if a subset index cannot be made without changing observation
   bytes.
3. For **each** target, run a watchdog-wrapped reconstruction using exactly
   `raw_boundary.cosine,projected_boundary.cosine,historical_a1.cosine,
   historical_a1_a2_anchor.cosine`, the pinned prototype/lens/model assets,
   the shared projected table, and the exact four-ID anchor declaration. The
   canonical chunk settings are query 256 and prototype 8192. The command
   shape is:

   ```text
   python3 scripts/trr_p03/resource_watchdog.py \
     --output-root <watchdog-qual-root> --cwd /tmp/trr-p03 \
     --timeout-seconds 3600 --max-rss-bytes 8589934592 \
     --min-available-bytes 10737418240 -- \
     env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src:scripts/trr_p01 \
     python3 scripts/trr_p03/reconstruct.py \
       --observation-index <truth-free-qual-index> \
       --prototype <pinned-boundary-table> \
       --historical-lens <pinned-lens> \
       --projected-prototype <single-prepared-projected-table> \
       --model-path <pinned-public-base-llama-snapshot> \
       --anchor-records <exact-anchor-json> \
       --methods raw_boundary.cosine,projected_boundary.cosine,historical_a1.cosine,historical_a1_a2_anchor.cosine \
       --output-root <reconstruction-output> \
       --query-chunk-size 256 --prototype-chunk-size 8192 \
       --plan experiments/TRR-P03/plan.json \
       --implementation-commit <frozen-commit>
   ```

4. Repeat the same truth-free qualifier for the second target, using the same
   pinned **public base** Llama model path. The evaluator-only generation step
   is where the matched and shifted target checkpoints differ; reconstruction
   must never receive the shifted checkpoint. Both target runs must pass this
   check, providing minimum evidence for the largest static length-128 cell
   plus the native four-record A1+A2 cell.

Use the declared canonical query/prototype chunks (256/8192) for the
qualification and the full matrix. If a resource workaround changes batching
or chunking, compare its numeric outputs and predictions with the canonical
run and retain/exclude the alternative according to exact output equivalence;
an unrequested alternative failure does not invalidate the canonical run.
The qualification itself never authorizes opening truth and does not replace
the complete matrix receipt. After both full 24-record prediction roots are
frozen, run the strict joint validator; only its PASS receipt authorizes the
Stage 1 score command.


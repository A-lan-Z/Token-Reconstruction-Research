# TRR-0010 stage-3 contract: remaining fields

This is a bounded handoff checklist, not a new proposal or execution schema. It
lists the fields that remain unset after the v4 proposal and the signed stage-1
agreement/amendments. Stage-1 public capture and public-base validation audit
are complete; no substantive fit, final source selection, or final-truth result
is claimed here.

## Already-bound context

- The shared starting point is the published TRR-0009 `continued_fixed_readout`
  state at selected step 400: commit
  `4602f98eeb03ac121d8bbc230d7e2b219551b914`,
  `experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors`,
  29,390,492 bytes, SHA-256
  `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`.
  This is the unchanged reference and the initialization for all four arms.
- The v4 design is in
  `experiments/TRR-0010/planning/shared_contract.proposal.json` (schema
  `token-reconstruction.trr0010-shared-contract-proposal.v4`). The signed
  stage-1 plan is
  `shared_stage1_public_bank_plan.json`, 26,127 bytes, SHA-256
  `bca93d6099760791c5ecbf3be8177d250cc672c62b2177a3d72d79b934c3826c`;
  its agreement is recorded in `joint_stage1_agreement.json`. The signed
  correction and compatible-selection supplement are
  `shared_stage1_correction_addendum_r1.json` and
  `stage1_template_compatible_selection_supplement_r1.json`.
- Count-only exclusion scanning is complete, but it is not source selection:
  `count_scan/inventory_7000_trr9_p08_union.json` (SHA-256
  `d64eae6b5e0137f1fb76e89ba7b618455853977330f13058877c6a7c3ab9038b`)
  reports 3,867 Finance and 236 Pile eligible unique records after the
  TRR-0009/P08 union. The failed 6,000-row Pile extension remains excluded.
- Capacity's current implementation/synthetic evidence is
  `setup/agent1_final_handoff_v2.json` (SHA-256
  `191f0b23d5a09bc830cc544dea17e07908dabef0cf7303333dc3b820b1532a54`),
  with nine tests recorded in `setup/model_tests_v6.json` (SHA-256
  `d46b8fcfa131ee4014d5d9086200aa0c96ea79bace8f36f7913be24dc762a0e9`).
  The serialized-W smoke passed, but it is not a largest-cell fit
  qualification.

## Fields to bind before fitting or final evaluation

### 1. Expanded bank, support, and exclusions — A2 / public-bank owner

Bind one immutable B0/B1 preparation receipt containing:

- B0's exact published 1,200-record payload and byte identity, including its
  124,371 published fit positions. B0 is reused unchanged; known old identities
  exclude only new B1 additions and the final evaluation panel.
- B1's exact output paths, bytes, SHA-256 values, record order/map digest,
  parent-source and final-sequence digests, and sidecar/metadata hashes. The
  target is 12,000 records (1,200 B0 plus 10,800 additions), with the signed
  strata proposal: B1 totals 6,000 Alpaca natural, 3,000 Pile natural, 1,800
  Finance natural, 600 Pile controlled, and 600 Finance controlled.
- The actual recipe and exclusion ledgers: known fit/validation/opened/TRR9/
  TRR8/P08 identities, approved opaque ledgers, parent-versus-derived-row
  semantics, and a zero-overlap receipt. The receipt must say whether the
  declared mixture was preserved; an all-natural append is reported as a
  coverage-plus-mixture factor.
- Actual supported vocabulary IDs and counts, support tensor/container
  hashes, per-stratum exposure counts, valid fitting-position count
  `actual_B1_valid_positions`, and exact sampler draw accounting. Distinguish
  unique parent sources, final sequences, valid positions, support rows, and
  repeated/derived variants.

A qualified B1 metadata manifest, support descriptor, and stage-1 capture
binding are now recorded: 12,000 records (1,200 unchanged B0 plus 10,800
additions), 1,243,710 valid positions, and support size 45,631, yielding
`N=13000` under the formula below. The raw shards and support payloads remain
external; shared forward-batch geometry qualification and the final capacity
settings are still pending, so this metadata binding does not authorize fitting
by itself.

### 2. Exact N, checkpoint grid, and four-arm schedule — root + A2 + capacity

Before any fit, bind a signed execution receipt with:

- the actual `actual_B1_valid_positions` and
  `N = max(12000, 1000 * ceil((5 * actual_B1_valid_positions / 512) / 1000))`;
  stop for contract revision if `N > 20000`, and use the same exact N for all
  four arms;
- the sorted, deduplicated checkpoint grid
  `[0, 1000, 2000, 4000, 8000, 12000, N]`, with earliest maximum on the
  384-record public-base development metric and earliest-step tie breaking;
- the already-agreed geometry and optimizer values: full width-192 fitting,
  width-128 natural evaluation with 127 scored post-BOS positions, batch size
  8, 512 position draws/update, seed 4010, AdamW, decoder LR 2e-4,
  directional LR 1e-4, weight decay 0, gradient clip 1.0, cosine schedule,
  and no staging/search/fallback schedule;
- the final directional regularization/optimizer implementation and support
  tensor shapes from the capacity-qualified merged-W runner. No sparse score
  implementation may become a fit, validation, or deployment fallback.

### 3. Public P0 development labels and diagnostic panel — A2/source owner + Agent1

Bind the public development observations used only for checkpoint selection:

- the exact 384-record panel (Finance 256, Pile 128), source IDs/order digest,
  final-sequence/H128 digests, public-base target/label or observation
  descriptor, and exclusion receipt proving that the panel and derived rows
  are outside B0/B1 and the final panel;
- each bank's 64-record fit-diagnostic subset, with quotas Alpaca 32, Pile
  natural 16, Finance natural 10, Pile controlled 3, and Finance controlled
  3; bind its IDs/order and sequence digest. Use the same indices for fixed
  and directional arms within a bank, rank with the frozen
  `TRR0010|fixed-diagnostic|4010|record_id` rule, and retain full fit width
  192. These subsets are diagnostic only and must not silently become
  checkpoint selection or final-truth data;
- explicit width-192 fit versus width-128 development geometry. LoRA is not a
  checkpoint-selection condition; selection uses public-base only.

The new A2 public-validation manifest and rows are externally prepared and the
compact audit receipt is now bound. It proves exact IDs, token IDs, masks,
positions, and H file hashes for all 384 public-base records without model,
LoRA, activation, or final-truth access. The inherited metadata reference is
retained for provenance; this remains checkpoint-selection evidence only.

### 4. Largest-cell resource qualification and caps — capacity + root

Supply the measured, fail-closed qualification receipt for the actual B1
support and merged full-W path:

- isolated per-method CUDA peak (with reset semantics), host RSS high-water,
  wall time, disk footprint, available-memory margin, and guard thresholds for
  the largest expanded-directional cell;
- exact agreed wall, disk, host-RSS, and isolated-GPU caps, plus the common
  preparation/export accounting and the warmed candidate/A1+A2 cost rule;
- before fitting, bind the base/support reload checks and resource guard; after
  fitting and before final evaluation, bind each produced state/W/support hash
  and the serialized output-equivalence checks, including the nonzero-Delta
  full-logit/argmax assertion. The deployed directional method loads a
  base-only decoder state plus one effective W; it does not retain E beside W
  or rebuild W per record.

The current 6.389/11.477 GiB estimates and the opened-fixture GPU diagnostic
are planning/smoke evidence only. They do not satisfy this qualification.

### 5. Four-arm runner, code bindings, and A1+A2 adapter — Agent1 + timing

Bind the exact executable source and training-input identities before fitting;
bind produced state/readout identities before final evaluation:

- full commit and hashes for the four-arm runner, gate, model/readout loader,
  registration adapter, and every loaded method/A2 module; code bindings must
  match the files imported at runtime;
- the TRR9 starting state above, B0/B1 inputs/support artifacts, and fixed
  public readout before fitting; after fitting, bind each arm's produced
  serialized state/readout hashes before evaluation. Directional inference
  uses the exported merged W directly; fixed arms use the declared public
  readout;
- the published A1+A2 adapter call: propose 512 candidates with chunk 256,
  retain the first 256 proposals before candidate simulations, and bind the
  public prefix/lens/reference assets plus an opened-fixture equivalence
  receipt. Do not relabel this as a native top-256 implementation;
- scorer registration of every active method under both B0 and B1, with the
  same per-cell registration/order and a truth descriptor that carries the
  registration hash, panel record IDs/order, input-bank hashes, and truth
  payload/order hash.

### 6. Final panel, contender freeze, and practical decision rules — root + A2 + Agent1

Before final source selection or truth, freeze the final evaluation design and
selected contenders, then bind:

- one 128-Finance + 128-Pile natural panel, paired under public-base and
  public-LoRA-2601 (512 target observations total, 127 post-BOS positions per
  clip), source IDs/order, final-sequence/H128 hashes, all known exclusion
  ledgers, and zero intersections with B0/B1/opaque reservations;
- the inherited B8x192 padded capture geometry and first-128 retention rule,
  plus the final truth descriptor/loader call. The single historical A1+A2
  comparator uses the same panel; it is one comparator, not separate A1/A2
  arms. Source records remain the bootstrap unit and domains/targets are never
  pooled;
- the final practical criteria from v4, without post-score edits: primary
  `expanded_directional` versus both `expanded_fixed` and `current_fixed`;
  token useful route requires 20% remaining-error reduction, a 0.5-point
  absolute floor where headroom permits, and positive paired lower bound;
  exact useful route requires 5 points and a positive paired lower bound;
  the separate data route compares expanded-fixed with current-fixed and the
  unchanged start; both routes report the predeclared 25% A1+A2 gap-closure
  rule and visible token/exact floors;
- the statistical and safeguard bindings: source-record bootstrap seed 10010,
  10,000 draws, benefit alpha .025, harm lower q .05, exact paired-discordance
  CP components (.0125 benefit, .025 harm), token/exact harm margins -0.5/-3
  points, rare bins 0/1-4/5-9/10-49/50+ with fewer than 32 exposed sources
  marked UNKNOWN/NO_EXPOSURE, ratio routes fail closed on nonpositive
  denominators, and all main-cell cost/harm gates. No pooled family claim or
  automatic promotion follows from a one-domain result.

The remaining prerequisites are the future final-evaluation forward-batch
qualification and capacity settings, capacity's measured largest-cell caps,
the exact four-arm/code receipt, and the root/A2 final-panel and criteria
freeze. Until those receipts are bound, this document authorizes no substantive
fit, final source selection, or truth opening.

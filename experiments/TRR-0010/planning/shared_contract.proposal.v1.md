# TRR-0010 shared contract proposal

**Status: proposal only.** No public-bank preparation, source selection, fitting, truth opening, or scientific freeze has occurred. This draft uses the published TRR-0009 checkout `4602f98eeb03ac121d8bbc230d7e2b219551b914` and starts all four arms from its selected competent fixed-readout state: `continued_fixed_readout`, step 400, 29,390,492 bytes, SHA-256 `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`. The unchanged TRR-0007 current-bank state remains a reported reference.

## Crossed design

The study crosses public-data coverage with readout freedom:

- current bank / fixed readout;
- current bank / directional readout;
- expanded bank / fixed readout; and
- expanded bank / directional readout.

The four arms receive the same optimizer update count, batch size, schedule, validation grid, and decoder continuation opportunity. Current fixed is the continued-training control for current coverage. Expanded fixed isolates the data-scaling factor. Expanded directional is the predeclared candidate. Report the directional-current, directional-expanded, fixed-data, combined-candidate, and two-factor interaction contrasts separately; keep Finance, Pile, and target conditions separate.

The proposed nested bank target is 1,200 unique current fit records and roughly 12,000 expanded records, including the current bank and at least 10,000 unique records overall. A2 must bind the actual inventory and construction before root freezes the design. The expanded bank should preserve the current mixture and domain/style strata where feasible. If preparation can only append natural rows, report the data factor as coverage-plus-mixture. Report unique parent sources, unique final sequences, valid token/context positions, identity support rows, derived variants, and repeated exposures as separate quantities; derived variants are not independent natural sources.

## Training and directional readout

The A2 runner proposal is 12,000 updates, with 512 position draws per update, batch size 8, AdamW, cosine decay, base learning rate 2e-4, weight decay 0, gradient clipping 1.0, and seed 4010. The exact update count is frozen once, before fitting, and used by all four arms. If the verified expanded valid-position count makes 12,000 updates provide fewer than six position exposures, increase the common count within the 12,000–20,000 bound; target six to ten exposures. The same fixed checkpoint grid is used for every arm: 0, 1,000, 2,000, 4,000, 8,000, and the final count. Curves use a fixed public diagnostic subset, while full-bank fit metrics are recorded at start and end.

The proposed directional adapter is a genuine token-specific low-rank residual over the public readout. For fitting-supported rows `S`, `E_prime[S] = E[S] + A @ B`, where `A` has shape `[|S|, 32]` and `B` has shape `[32, 2048]`; rows outside `S` remain the public readout. `A` starts at zero and `B` at a deterministic public orthonormal basis. Both are trained under the same optimizer opportunity as the decoder. A weighted `1e-4` L2 penalty on `A` and a `1e-4` penalty on `B-B0`, plus a proposed per-row update norm cap of `0.5 × ||E_v||2`, are pending capacity review. The support set comes from fitting data only. Full-vocabulary current-H-only argmax remains the deployment path, with no routing, search, staging schedule, teacher ranking, or A2 fallback. Unsupported rows must remain tensor-equivalent to the public anchor, and synthetic/public diagnostics must show non-collinear token-row updates.

Fitting uses a separately declared 192-width context geometry. Natural validation and final clips use width 128 and score all 127 post-BOS positions. Capacity must verify the reported 192-versus-128 crop distinction before freezing; no geometry substitution is allowed.

## Validation and final panel

Checkpoint selection uses the opened TRR-0009 matched-public natural panel as TRR-0010 public development only: 256 Finance and 128 Pile records, paired over `public_base` and `public_lora_2601` (384 unique records, 768 target observations). The selection metric is the equal-domain-weighted natural token accuracy, with earliest maximum on the fixed grid. A2 must bind record and sequence hashes, and the panel plus derived rows are excluded from both fitting banks and the final natural panel. The inherited 48-record style-balanced panel is secondary diagnostic context only.

The proposed one-shot final natural panel contains 256 unique records: 128 Finance and 128 Pile, paired over the same two target conditions (512 target observations). This is the upper end of the resource-dependent 64–128/domain common-subset range. A1, A2, all four TRR-0010 arms, and the unchanged reference use the same panel; the source-record denominator is 256, with target observations nested within source. All source identities, fit/validation roles, method outputs, and exclusion ledgers are frozen before the one truth opening. A2 is evaluated as a comparison only and is never included in deployed inference.

## Useful outcome and cost proposal

The primary candidate is expanded directional. It must clear both fixed controls, expanded fixed (matched data) and current fixed (continued-training control); there is no posthoc fixed-arm winner. A useful route must clear both an absolute threshold and a baseline-to-A2 gap-closure threshold. For token accuracy, require at least +1.0 percentage point against each fixed control and a one-sided 97.5% paired lower bound above zero. For exact clip recovery, require at least +5.0 points against each fixed control with the same lower-bound requirement. The same metric must close at least 25% of a visible A1-to-A2 gap: a visible gap is at least 1 token point or 5 exact points, and the one-sided closure lower bound must be at least 25%. If A2 does not exceed A1 or the gap is below its visibility floor, closure is `UNKNOWN`; it cannot support promotion. Exact and token routes use alpha 0.025 each, with source-record bootstrap draws, no pooled domains or target conditions, and explicit denominators.

Every cell must pass the −5 exact-point and −1 token-point harm safeguards. Rare/absent frequency bins use a 32-exposed-source floor and a one-sided −1 token-point lower bound; insufficient support is `UNKNOWN`. Proposed cost limits are 1.50× fixed warmed runtime, 1.50× fixed peak GPU memory, at most 256 MiB extra serialized candidate state, and at most 0.50× A2 warmed runtime. The fit budget is at most 20,000 equal updates per arm, with shared public preparation charged once and no paid compute. The published TRR-0009 peak is a qualification reference, not a guarantee; the largest expanded-directional cell and serialized adapter must pass an isolated resource and output-equivalence qualification before the full matrix.

This proposal is not yet a common agreement. It remains pending A2’s exact bank inventory and mixture construction, 192-width fitting geometry and exposure calculation, one frozen update count, directional implementation/resource review, A1/A2 comparator identities, and final 128-per-domain panel availability and exclusions.

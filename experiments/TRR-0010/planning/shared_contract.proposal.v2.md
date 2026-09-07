# TRR-0010 shared contract proposal v2

**Status:** proposal v2, pending root review before A2 handoff. This document records a proposed common contract; it is not a scientific freeze, source selection, fit, truth opening, or promotion decision.

## Starting point and crossed design

All four new arms start from the same published TRR-0009 `continued_fixed_readout` checkpoint selected at step 400: SHA-256 `5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14`, 29,390,492 bytes. That checkpoint is both the unchanged shared starting reference and the baseline for the baseline-to-A1+A2 comparison. The historical A1+A2 result is one comparator, not separate A1 and A2 methods; there is no new A1-only arm.

The primary crossed design is public-bank scale by readout treatment:

| Bank | Fixed readout | Directional readout |
|---|---|---|
| Current public bank | current-fixed | current-directional |
| Expanded public bank | expanded-fixed | expanded-directional |

The directional arm tests learned token directions while keeping token identities and the decoder geometry fixed. The primary candidate is expanded-directional. Expanded-fixed is the matched data-use control, and current-fixed is the strongest directly applicable continuation control. The study reports the 2x2 data, readout, and interaction effects; a flat directional interaction does not invalidate the separate expanded-data route.

## Bank and fit proposal

The current bank target is 1,200 fitting records. The expanded target is 12,000 records, with a minimum of 10,000 and at least 8,800 genuinely new records relative to the current bank. The bank construction recipe and mixture should be preserved where feasible and reported explicitly if expansion adds natural-only material. Parent sources, derived variants, and controlled replacements are not independent natural sources. If controlled replacements are used, the report must identify at least 128 such records and account for them separately. A2 has proposed, pending a bank manifest and root freeze, a current 1,200-record stratum of 600 Alpaca, 300 Pile-natural, 180 Finance-natural, 60 Pile-controlled, and 60 Finance-controlled records. Its exact 10x expansion proposal adds 5,400 Alpaca, 2,700 Pile-natural, 1,620 Finance-natural, 540 Pile-controlled, and 540 Finance-controlled records. An all-natural append would change the controlled share from 10% to about 1% and is therefore rejected as a mixture confound; unique new-parent availability and public-cache capacity remain open. The pending B0 list has 120 controlled records, which A2 must reconcile with the separate requirement to report at least 128 controlled replacement draws before freeze.

The provisional common schedule is 12,000 updates for all four arms. Each update draws 512 token positions, or 6,144,000 position draws across the run. The expanded-bank estimate is about 1.24 million positions, giving about 4.94 exposures per position; the practical gate is at least 5 exposures, with a target range of 5–10. A single common rounded update count may be increased, up to 20,000, only if the validated expanded-bank count shows fewer than five exposures. This is a coverage guard, not a staged training objective.

Fit context width is 192. The inherited width-192 schedule stores positions 1 through 191, including 131,576 positions 128–191 among 1,536,000 historical draws; this is full fit geometry, not a suggested crop. Natural validation uses width 128 and must be declared separately. The fixed checkpoint grid is a common evaluation grid such as steps 0, 1,000, 2,000, 4,000, 8,000, and 12,000. A fixed public diagnostic subset may be used for curves, with full-bank metrics at the start and end; full expanded-bank scoring at every checkpoint is not required.

## Directional readout and deployment accounting

The preferred directional parameterization is capacity-owned full independent supported-token rows:

`E_eff[v] = E[v] + Delta[v]` for supported rows `S`; `E_eff[v] = E[v]` outside `S`.

`Delta[S]` is a `[|S|, 2048]` FP32 table, zero-initialized from the public embedding matrix. For the current 17,126-token support this is 35,074,048 learned parameters (140,296,192 parameter bytes; about 561,184,768 bytes for parameters, gradients, and Adam state). The all-vocabulary support bound is about 4.20 GB for that package before the published base peak. There is no hard gain/bias bound and no row renormalization. Unsupported rows have no correction parameter and remain exactly `E`; decoder competition can still change predictions.

The proposed fixed anchor is the frequency-weighted relative-L2 penalty `lambda * mean_v[(1 + 4/sqrt(count_v)) * ||Delta[v]||_2^2 / (H * rms(E[v])^2 + eps)]` with `lambda=1e-4`. Learning rate and optimizer details remain capacity-owned, while all four arms receive the same declared optimizer opportunity. Capacity must qualify the largest actual support against the approximately 12.2 GiB reserved-GPU envelope. Sparse score hooks are acceptable only if explicitly qualified against a materialized merged `W`; merged-W scoring and inference define the comparison path.

A deployed full-readout artifact `W` replaces the shared embedding artifact `E`; it is not charged as `E + W`. Training checkpoints and optimizer state are reported separately from the serialized deployed state. The resource report must distinguish serialized files, tensor payloads, process peaks, and pooled four-arm harness measurements.

## Public validation and final evaluation panels

Checkpoint selection uses only the 384 matched-public `public_base` observations: 256 Finance and 128 Pile. It uses the common domain-balanced token metric with an earliest-max rule. LoRA observations are not used for checkpoint selection, and the older 48-style panel is secondary context only. The matched validation records, their sequences, parent sources, and derived rows are excluded from subsequent bank additions and final evaluation.

The proposed final panel is 512 unique natural records per domain, 1,024 unique records total, with two target observations per record. Its exact size remains subject to source inventory and timing. A separate nested A1+A2 anchor panel is proposed at 128 unique records per domain (256 unique, 512 target observations), with a resource-dependent range of 64–128 records per domain. The anchor estimates historical gap closure; it does not replace the full final panel. Both domains are reported separately, and paired source records—not target observations—are the bootstrap unit.

## Prospective decision routes

The primary route is expanded-directional versus expanded-fixed, with robustness against current-fixed. A separate data-useful route compares expanded-fixed against current-fixed and the unchanged shared starting reference; it can succeed if the directional interaction is flat. The report must retain the 2x2 interaction and must not choose a favorable control after seeing results.

For each domain, a useful token result requires at least a 20% reduction in remaining token errors, at least +0.5 percentage points of accuracy against the relevant fixed control, and a positive source-record paired lower confidence bound. If the relevant fixed cell leaves less than 0.5 percentage points of attainable headroom, report that absolute floor as ceiling-limited rather than as a failure; the relative-error reduction, positive paired bound, and predeclared A1+A2 anchor gap-closure evidence still govern a useful claim. A useful exact result requires at least +5 percentage points exact accuracy and a positive paired lower bound. These are separate domain requirements; a one-domain success is reported as domain-specific and does not support a global promotion claim.

On the A1+A2 anchor, gap closure is measured from the unchanged shared starting reference toward the single historical A1+A2 comparator. It is assessed only when the observed gap is visible (at least 1 token percentage point or 5 exact percentage points). The candidate must close at least 25% of that gap with a paired lower bound supporting the same direction. If the comparator is no better than the unchanged reference or the gap is below the visibility floor, gap closure is unknown rather than zero.

Proposed safeguards require every active cell to stay within −3 exact percentage points and −0.5 token percentage points of its matched fixed control, with the same paired confidence procedure. Rare and absent frequency strata additionally require at least 64 exposed source records and a lower bound no worse than −0.5 percentage points; unsupported strata remain unknown. These harm limits and all confidence-tail choices remain pending root review.

## Cost and resource gates

Before a four-arm run, capacity must qualify the largest representative directional cell. Provisional gates are candidate-to-control warmed runtime ratio at most 1.5, pooled process-memory ratio at most 1.5, and directional-vs-fixed runtime increment at most 0.5. The training budget is at most 20,000 common updates per arm, with shared preparation charged once. No paid compute or unstated serialized-memory cap is assumed. A2 owns the public-bank preparation and common runner proposal; its reviewed handoff is `/tmp/trr-p09-fixed-control-review.md` (SHA-256 `d0ddc1ccbb9a247b59ecc10fd944bff9311584831283610975e784d0b289a75f`).

## Open items before agreement

Root must review and either freeze or revise the bank mixture, exact source and sequence bindings, validated exposure count, 192-versus-128 geometry, common update count, full-row capacity result, optimizer and regularization details, anchor/full-panel sizes, confidence tails, and cost gates. A2 must bind the matched-public validation record and sequence ledgers and exclude those records and derived rows from expansion. Until those items are agreed, all thresholds, counts, and parameterization choices above are proposals only.

The preserved first draft remains at `shared_contract.proposal.v1.json` and `shared_contract.proposal.v1.md`; it records superseded alternatives and is not the canonical proposal.

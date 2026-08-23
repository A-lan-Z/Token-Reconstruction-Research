# TRR-0002 owner-directed historical-input bridge revision R4

Date received: 2026-08-23 (Australia/Sydney)

The repository owner directed a correction to the heavier-target study so the
comparison changes only target weights while retaining the old evaluation.

Verbatim owner direction:

> Right, I see.
>
> So you changed the evaluation and the experiment, and now I have no idea what
> the new experiments mean when comparing them with the old ones.
>
> How do you think you can fix it?

After the proposed controlled bridge design, the owner authorized execution:

> Go ahead

The required bridge uses the exact 128 historical Finance inputs, their frozen
token IDs, 13,990 post-BOS scoring positions, cut depth, and historical metric.
It changes only target weights across the untouched public checkpoint, the
original Finance target, and the verified heavy full-SFT target.

Every compared reconstruction configuration must use identical public A1/A2
resources in every target cell. The historical A1 must be explicitly labelled
as containing the public-Alpaca fitted affine lens; the checkpoint-only A1 is a
separate method. No target prefix weights or target truth may enter reconstruction.

Run the historical adaptive baseline and the recent fixed and adaptive A1+A2
configurations under the exact old token-accuracy rule. Reproduce the original
Finance-target result before accepting the bridge. Report target-only paired
deltas, runtime/cost, and complete-input recovery as a secondary metric.

The existing GrandMaster results remain an auxiliary different-input robustness
panel and must not be compared directly with 98.22% as if only target weights
changed.

This revision continues the open TRR-0002 task on task/TRR-0002 and updates
pull request #3. It does not authorize merging or starting TRR-0003.

RESEARCH_CHARTER.md remains the sole authoritative scientific definition.

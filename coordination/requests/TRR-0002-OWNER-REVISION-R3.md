# TRR-0002 owner-directed strict-surrogate and heavier-target revision R3

Date received: 2026-08-23 (Australia/Sydney)

The repository owner directed two related extensions of the target/surrogate
transfer study:

1. remove the arbitrary public-Alpaca fitted lens from the primary surrogate
   condition, use only the untouched public checkpoint for A1 and A2, and
   report intuitive whole-input reconstruction metrics in addition to token
   accuracy; and
2. run another target-transfer experiment against a substantially more heavily
   fine-tuned public derivative whose documented base is the same official
   Llama 1B Instruct checkpoint.

Verbatim owner direction:

> The problem I have with this is because it's arbitrary. I can't just put these numbers in my report and explain that the A1 is using an Alpaca-fitted lens because we had it there already, so we just chose to use it.
>
> Does that make sense?
>
> And apart from the accuracy, I would also want to see, like, how many Input sentences, was it able to reconstruct perfectly? So what I want is a more intuitive metric that actually relates closer to how it will perform in its actual task.

> Yes, this makes sense. Apart from this, can you also do another round of experiments where the target is some Llama-3-1B-Instruct model that has undergone heavier fine-tuning, rather than just the finance fine-tune? For example, you may find one popular fine-tuned version online.
> Just make sure that it is actually fine-tuned on the Llama-3-1B-Instruct model, rather than something else.

This revision continues the still-open `TRR-0002` task on `task/TRR-0002` and
pull request #3. It does not authorize merging the pull request or starting
TRR-0003.

Model lineage must be verified from primary public metadata and pinned to an
exact revision. The primary strict-surrogate arm may use only weights and
computation contained in the untouched public base checkpoint; the historical
Alpaca-fitted lens is retained only as an explicitly labelled control.

The result must prominently report exact complete-input recovery (token-exact
records and decoded-text-exact records), token accuracy, errors among failed
records, length-stratified behavior, runtime, and candidate evaluations. Any
retrospective or non-blind evidence must remain labelled as such.

`RESEARCH_CHARTER.md` remains the sole authoritative scientific definition.

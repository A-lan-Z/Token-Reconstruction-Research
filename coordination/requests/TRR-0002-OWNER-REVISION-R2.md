# TRR-0002 owner-directed target/surrogate transfer revision R2

Date received: 2026-08-23 (Australia/Sydney)

The repository owner identified saturation in the public configuration
evaluation and directed a transfer evaluation in which the observed target
comes from a Finance-Instruct-fine-tuned Llama 3 1B Instruct model while A1 and
A2 retain only the corresponding untouched public model and public auxiliary
state.

Verbatim owner direction:

> One issue I see from the results is that the evaluation has clearly been saturated. It no longer differentiates the top candidates very well
> So, are you aware that this token reconstruction method is building on a public surrogate that is supposed to be similar to the actual target?
>
> For example, the target might be the Llama-3-1B-Instruct model, fine-tuned on several datasets, while the models we use for A1 and A2 are just the public Llama-3-1B-Instruct model without the additional fine-tuning.
> Can you try setting the target as the Llama 3 1B instruct model, fine-tuned on the finance instruction dataset? We should already have this model ready in our project.
>
> For the model for A1 and A2, let's just use the public base model.
> Does this make sense?

This revision continues the still-open `TRR-0002` task on `task/TRR-0002` and
pull request #3. It does not authorize merging the pull request or starting
TRR-0003.

The existing historical Finance setup already contains the requested model
separation: its layer-4 observations come from the generation-300
Finance-Instruct victim (`victim_post_000299`), while the A1 proposal and A2
causal simulation use the pinned untouched public
`meta-llama/Llama-3.2-1B-Instruct` checkpoint. R2 therefore evaluates a frozen,
diverse subset of the already-publicly-ranked configuration finalists on that
actual Finance target rather than creating a scientifically duplicate setup.

Because the historical Finance truth was opened before R2, the new comparison
is retrospective stress evidence. Predictions must nevertheless be generated
without a truth input, frozen and hashed before a separate scoring process
loads truth. The result may diagnose configuration differentiation under
target/surrogate mismatch, but it may not be relabelled as fresh blind evidence
or silently replace the previously frozen winner.

`RESEARCH_CHARTER.md` remains the sole authoritative scientific definition.

# TRR-0007 capacity phase design

## Question and crossed cells

This phase compares the retained TRR-0006 enriched trained-diagonal
positionwise decoder with one additional positionwise capacity extension.  The
four public fitting cells are:

| fit bank | current-family control | residual MLP extension |
| --- | --- | --- |
| retained TRR-0005 enriched bank | train the neutral diagonal state | train the same state plus a 2048 -> 512 -> 2048 GELU residual |
| TRR-0007 improved bank | train the neutral diagonal state | train the same state plus the same residual |

The bank interface is the TRR-0005 public-fit manifest contract: padded
`[records, positions, hidden]` activations, public token IDs, a right-padded
mask, record metadata, and the shared normalized public embedding table.  A
shared validation manifest is supplied to each bank.  Fit labels are public
training supervision; no final or sealed target resources are loaded.

## Initialization and decision rule

The published state at
`experiments/TRR-0005/joint_fit_v1/enriched/affine_trained_diagonal_attention128/selected.safetensors`
is byte-pinned as the frozen evaluation reference (`696eb9fc951e...d599a2`)
and is verified/recorded by the trainer.  The crossed training cells use a
common neutral TRR-0005 initialization instead: identity `W`, zero `b`,
`s=3`, deterministic diagonal Q/K/V, and zero output correction.  This avoids an unmeasured warm-start advantage on the old enriched bank; the
retained state's selected-step fit accuracy is measured separately as a frozen
reference diagnostic.  The
control loads that neutral diagonal state, while the extension copies it into
its base.  The extension's MLP down projection uses deterministic
Kaiming-uniform initialization and its up weight and bias are zero.
Consequently the two arms have identical step-0 projected rows and logits on
every valid position; the residual is a real trainable path after the first
update.  All base and MLP parameters are trainable in both arms, with the
same AdamW schedule, gradient clip, draw schedule, and validation checkpoint
rule.  The retained state is evaluated only as a public-fit diagnostic and
remains the separate frozen state supplied to fresh evaluation.

The extension consumes the current activation at each position only.  Its
nonlinear input is a fixed per-position layer normalization of `H_i`; no
earlier activation, source token, token history, candidate list, teacher
score, public-prefix call, or A2 route is available.  The output is added
before the inherited normalization and tied full-vocabulary projection.  The
embedding table remains a fixed public resource.

Checkpoint selection is the earliest maximum style-balanced token accuracy on
the common public validation split, including step zero.  In addition to the
full fit metrics, the runner freezes a deterministic challenge subset from
public fit positions that the common neutral state gets wrong at step zero.
It reports each checkpoint's accuracy/loss on that subset; the subset cannot be
perfect at initialization by construction.  Challenge metrics are descriptive
optimization/capacity diagnostics and are not used to select a model.

## Resource forecast and bounded execution

At the representative geometry (`B=8`, `T=192`, `D=2048`, `K=512`,
`V=128256`, `M=512`), the extra MLP has
`2*(2048*512) + 512 + 2048 = 2,099,712` trainable parameters, or about 8.0
MiB in FP32 before optimizer state.  The inherited diagonal state has about
5.25M parameters.  The selected full-vocabulary logits buffer is
`512*128256*4 = 262,668,288` bytes (250.5 MiB); backward keeps additional
logit/autograd workspaces.  The resident normalized embedding table is about
1.05 GB on disk and is loaded once per bank.  The preflight therefore budgets the existing measured TRR-0005 1.5x floor
(4,413,456,384 bytes) plus the MLP activation/parameter and optimizer
allowance, and records live host/RSS guards before every public load and
update.

The first run is a bounded qualifier on the full representative gathered
geometry (B=8, T=192, D=2048, K=512, V=128256) and a short step prefix.  It
validates state loading, step-0 equivalence, current-H isolation, finite
gradients, checkpoint serialization, and challenge-subset progress.  The unit
tests use a reduced synthetic fixture only for control-flow checks.  No GPU or
heavy-memory fit is launched until the orchestrator records explicit resource
clearance.  Full cells use create-only output roots,
deterministic seeds, retained schedules, per-stage timings, peak memory,
command/environment metadata, and preserved failure receipts.

## Handoff interface

`src/token_reconstruction/trr0007_positionwise.py` provides the model,
retained-state loader, state serializer, projected-row/logit interface, and
step-0 equivalence helper.  `scripts/trr0007_train_positionwise.py` provides
the manifest loader adapter, resource preflight, deterministic schedule,
crossed control/extension training, challenge diagnostics, and task-local
receipts.  Evaluation remains in the separate TRR-0007 evaluation runner and
will consume only the selected state descriptors plus the public embedding
table and fresh activation observations.

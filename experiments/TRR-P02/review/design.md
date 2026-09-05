# TRR-P02 scientific design

Status: independent design review, prepared after the P02 setup packet was
saved. This file is the bounded public diagnostic plan; it is not a
reconstruction result.

## Decision question

The first decision is whether P01's position-two collapse is an execution
problem or a failure of the fixed-reference/shared-offset model. The matched
public Llama checkpoint is sufficient for that decision. A shifted target or a
new evaluation panel stays out of scope until this public mechanism check
survives.

The checkpoint is `meta-llama/Llama-3.2-1B-Instruct`, revision
`9213176726f574b556790deb65791e0c5aa438b6`, with layers 0--3 (cut 4), BOS
128000, hidden width 2048, model-native BF16 inference, `eval()`, and
inference mode. Reuse the P01 full-vocabulary table read-only from
`experiments/TRR-P01/runtime/cpu-table-20260905/boundary_prototypes.safetensors`
(SHA-256
`51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3`). No
context-by-vocabulary activation table is built. The fixed public diagnostic
seed is 314159; it selects no source records and opens no private truth.

The declared token panel is `T=(13,32,198,220,2048,4096,16384,29871)`.
Use reference `r=220`, and use `v=4096` and `w=2048` for the fixed endpoint
and pair controls. Every ID is a known public token ID. The runner's context
specifications include BOS explicitly:

| context | public token IDs | endpoint position | role |
| --- | --- | ---: | --- |
| C0 | `[BOS]` | 1 | baseline `b_t=z(C0,t)` |
| C1 | `[BOS,13]` | 2 | equal-length primary context; repeated-panel length 1 |
| C2 | `[BOS,198]` | 2 | equal-length primary context |
| C3 | `[BOS,1024]` | 2 | equal-length primary context |
| C4 | `[BOS,29871]` | 2 | equal-length primary context |
| C5 | `[BOS,13,13]` | 3 | repeated-13 context-length control |
| C6 | `[BOS,13,13,13]` | 4 | repeated-13 context-length control |

The primary panel is C0--C4 with all eight endpoint IDs: 40 rows. The
repeated panel reuses C0 and C1 and adds three endpoint IDs (`r,v,w`) at C5
and C6: 46 unique activation rows total. C1--C4 are endpoint-position
matched at position 2. C5/C6 change the visible prefix and all endpoint
relative positions together, so they are labelled context-length/relative-
context controls; they are never presented as a pure absolute-position test.

For every declared context/token pair, define

```text
z(C,t) = public_prefix(C + [t])[-1]
b_t    = z(C0,t)
d_t(C) = z(C,t) - b_t
```

Here every C is a **public teacher-prefix control**: all IDs are known by
construction and none is a hidden prefix from an evaluated reconstruction.
Keep these exact teacher-prefix offsets separate from offsets obtained after
any simulated reconstructed prefix. If a correction fails with exact public
C on `[BOS,u,v]`, the P01 wrong-prefix cascade is ruled out for that failure.
If it succeeds only with exact C, prefix error remains a possible contributor.

## Wiring and cache gate

Run this gate before interpreting deformation or ranking statistics.

1. For the tiny matched sequence `[BOS,u,v]` with `u=13` and the declared
   endpoint `v=4096`, compare one native `forward_full` call with a fresh
   cache receiving BOS at `start_pos=0`, `u` at 1, and `v` at 2. Compare the
   hidden output at **every sequence position**, including the final endpoint;
   record maximum absolute error, RMS error, cosine, positions, and cache
   lengths after each commit. Also record the final endpoint's P01-table
   top-1, true-token rank, and top-two margin for full and cached outputs.
   Repeat the endpoint check with the fixed reference `r` and pair token `w`
   only where it is already part of the declared public panel. A small batch
   of the four one-token contexts plus one serial case checks that batching
   does not hide a position offset. A discrepancy is reported at its measured
   scale rather than called a context effect.

2. Check the fixed-reference index and sign on the exact position-two
   public prefix C1 (`[BOS,13]`, yielding `[BOS,13,r]`), using

   ```text
   b_r       = z(C0,r)
   o_r(C1)   = z(C1,r) - b_r
   q_minus   = z(C1,r) - o_r(C1)
   q_plus    = z(C1,r) + o_r(C1)
   ```

   `q_minus` must equal the recomputed `b_r` to the recorded arithmetic
   tolerance and should rank `r` first against the fixed baseline dictionary
   (with exact ties recorded under deterministic ID ordering). `q_plus` is an
   opposite-sign wiring control only. This catches a reversed offset, a
   prototype/last-position mismatch, and a wrong reference position before
   any hidden-token interpretation. The repeated C5/C6 rows remain context
   length controls; they do not add another sign arm.

3. Check probe isolation. From a persistent cache containing each declared
   public C, clone the cache and append `r` only to the clone. Then append a
   selected public endpoint to the original and compare that output with an
   independently constructed cache for the same public prefix plus endpoint.
   The original cache length must remain `len(C)` before its selected commit.

RoFormer defines rotations of query/key representations whose dot product
depends on relative position (`m-n`) and whose rotations preserve vector norm;
see [Su et al., RoFormer, arXiv:2104.09864](https://arxiv.org/abs/2104.09864),
Secs. 3.1--3.2 and Eq. 35. The attention operation still mixes the visible
causal context through content-dependent weights, as defined by scaled
dot-product attention in [Vaswani et al., Attention Is All You Need,
arXiv:1706.03762](https://arxiv.org/abs/1706.03762), Sec. 3.2.1. Thus a
uniform position translation can be a numerical q/k check only; it is not a
surrogate for adding prefix tokens, and no inverse residual RoPE rotation is
applied to observations.

## Shared-offset and ranking panel

For C1--C4 and all `t` in T, collect the endpoint activations in one reused
public panel. Keep C0 as the explicit zero-offset baseline. For each C1--C4,
compute

```text
dbar(C) = mean_t d_t(C)
R(C)    = median_t ||d_t(C)-dbar(C)||_2 / median_t ||d_t(C)||_2
```

and report the median cosine of each `d_t(C)` with `dbar(C)`, handling zero
vectors explicitly. C0 is shown as a baseline row but excluded from the
C1--C4 decision aggregates so its zero offset cannot dilute the result.

For token-pair deformation, use `(v,w)` and two fixed additional pairs from
T, declared before execution. For each C1--C4 report

```text
e_C(v,w) = [z(C,v)-z(C,w)] - [b_v-b_w]
```

with its L2 norm, ratio to `||b_v-b_w||`, and pair cosine to the baseline
pair. C0 rows are retained only as the explicit zero reference. This directly
tests whether context changes token-identity directions rather than adding a
shared vector.

For every primary C1--C4 query, rank its known public token t under the same
cosine normalization and fixed baseline dictionary in three labelled arms:

* raw `z(C,t)` against `b_T`;
* P01 reference subtraction `z(C,t)-o_r(C)` against `b_T`;
* common subtraction `z(C,t)-dbar(C)` against `b_T`.

Report top-1, true-token rank, and `s_true - max_{q != t}s_q`, including the
runner-up margin. The `dbar` arm is an oracle/public-panel centering
 diagnostic because the known public token panel supplies its context mean;
it is not a permissible no-fit reconstruction method.

To retain nearby directions that may matter for P01 errors, derive each
declared token's fixed local dictionary once from the read-only P01 table:
the token itself plus its eight nearest **other** baseline prototypes (N8),
with cosine and deterministic token-ID tie breaking. This gives nine entries
per local dictionary and keeps the true token explicit. Run the same primary
queries in those local dictionaries and in the fixed T dictionary. These are
restricted-dictionary controls and are never reported as full-vocabulary
quality.

Predeclare exactly one targeted full-vocabulary lookup: C1--C4 crossed with
`{r,v,w}`, 12 rows. Use the same public baseline dictionary for raw,
mean-centered oracle, and reference-subtraction ranks, and report full-V
rank/margin beside the restricted controls. Do not adaptively add queries or
write a per-context 128k table.

For the repeated-13 panel, reuse only `{r,v,w}` at C0/C1/C5/C6. Record
those endpoint activations, their endpoint positions, and the bounded
same-token/different-token separation summary, with C0/C1 rows identified as
reused primary rows. This supplies the context-length/relative-position check
without adding full-vocabulary rows or another ranking arm.

## Frozen-lens geometry diagnostic

Load the exact read-only public historical affine lens if available:
`/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/inversion_20260730/out/lens_alpaca.pt`,
SHA-256
`33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`.
Define `g(h)=W h+b`, freeze it in eval mode, and fit nothing in this task.
Use the same cosine normalization on query and candidate rows and the same
candidate sets for all comparisons:

1. raw boundary: `normalize(z(C,t))` against `normalize(b_q)`;
2. projected prototypes: `normalize(g(z(C,t)))` against
   `normalize(g(b_q))`, using one streamed projection of the reused P01 table;
3. ordinary historical A1: `normalize(g(z(C,t)))` against the normalized raw
   public input embedding `normalize(E_q)`.

Compare true-token rank and cosine margins on the same 12 targeted full-V
rows for all three arms, so raw/projection/A1 use one identical candidate
dictionary. FrozenAffineLens.forward returns cosine logits multiplied by
`exp(s)`: preserve native A1 scores/ranks and record `s`/`exp(s)`, while
reporting cosine-equivalent A1 scores and margins after division by the positive
scale for cross-arm comparisons. The local N=8 and T results remain the
fitting-free restricted-dictionary controls. All rank arms use strict rank
`1 + count(score > true_score)` with equal-score counts recorded separately;
stable score-descending/ID-ascending ordering selects top-1 and runner-up IDs.
Also compare same-token cross-context variation against different-token
within-context separation in raw and projected primary C1--C4 geometry. The
projected-prototype arm and A1 arm are **fitted-lens diagnostics**, not no-fit
methods. A lens improvement only shows that this fixed public map makes these
queries easier to separate and may guide a learned-decoder workstream; it does
not establish that external context or position features are required. That
claim needs a same-activation feature comparison.

If the lens is unavailable, record the exact limitation and complete the
public-context panel without substituting a new fit.

## Decision and stopping rule

Advance one deterministic multi-reference mean-offset follow-up only if the
cache/sign gates pass and the equal-length C1--C4 measurements jointly show a
common component: low C1--C4 residual-to-delta ratios, small pair deformation
relative to baseline token separation, and consistent true-rank/margin gains
for the public oracle or reference correction. Such a follow-up would be a
variance-reduced estimate of the measured public shared component, with all
reference IDs declared before execution.

If cache/full or self-consistency fails, repair the wiring before a geometry
claim. If those gates pass but token-pair deformation is comparable to token
separation, or correction gains are inconsistent, stop and deprioritize the
static dictionary family. Do not add references, thresholds, a larger table,
or a broad context sweep in this round. A fitted-lens improvement does not
make the no-fit prototype method valid and does not by itself prove that
context or position features are needed.

## Resource and provenance budget

This is CPU-only, one isolated process after the exclusive slot is acknowledged.
P01 measured about 5.00 GiB peak RSS for the full reconstruction process and
used an approximately 501 MiB prototype payload. Reusing that table, streaming
one frozen-lens projected vocabulary table, and limiting full-V work to 12
queries gives an expected 6--8 GiB peak on the 30 GiB host. Reserve 8 GiB RSS
for the process, retain a 10 GiB available-host-memory fail-closed guard, and
retain at most one 8.2 MiB float32 full-V score buffer per query chunk. The
expected runtime is roughly 180--480 seconds including model/table/lens I/O.
No GPU allocation, paid compute, private target, source text, target weights,
or TRR-0004 artifact is required. Record commit, packet hash, model/tokenizer
identities, exact token/context lists, seed, device, timing, peak RSS, query
counts, hashes, and excluded controls in the manifest. Add this design note to
the final publication inventory and manifest provenance.

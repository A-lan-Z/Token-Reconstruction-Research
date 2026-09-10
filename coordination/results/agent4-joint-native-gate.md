# Agent 4: bounded native joint-activation proposal gate

**Decision: stop this native window configuration.** Joint-observation proposals
included neither previously missing root token. Wider ordinary sequential search
also found neither root, but used less measured search time and calibrated work at
each position. The supplied stop conditions are met. No larger study was launched.
This is a negative result for the tested component, not an impossibility result for
joint evidence, fitting-free inversion, or other proposal mechanisms.

## Executed result

The two opened diagnostics are `natural_11_1:9` and `natural_84_1:18`, with BOS at
position0. Histories were loaded from the original frozen predictions before each
root; the separate evaluator confirmed both histories were correct. The runner
never loaded source labels, supplied true future IDs, or inserted a true candidate.
All six lists and outputs were frozen and hashed before this study's scorer read
the previously opened truth. No extra diagnostic positions were selected.

| Native component | Correct roots proposed | Verified | Returned | Search seconds, both roots | Calibrated work seconds |
|---|---:|---:|---:|---:|---:|
| Joint observations, window3 | 0/2 | 0/2 | 0/2 | 0.439784 | 0.348439 |
| Own-observation proposals, same block verification | 0/2 | 0/2 | 0/2 | 0.201218 | 0.325578 |
| Wider ordinary sequential, width16 | 0/2 | 0/2 | 0/2 | 0.246172 | 0.235222 |

Joint and diagonal proposed the same two root IDs at each root, including the
incumbent. Their first common-state proposals were identical at all three slots.
None of their root IDs was new relative to the original65 checked candidates.
Wider sequential proposed33 and36 distinct root IDs, including24 and30 absent
from those original lists, but none was correct. Newly included correct roots,
lost correct proposals, and useful correct proposals per measured second are all
zero. No arm achieved strict numerical acceptance. Proposed, actually verified,
and returned IDs are recorded separately in the six frozen JSON files.

| Opened root | Joint search seconds | Wider sequential seconds | Joint termination |
|---|---:|---:|---|
| natural_11_1:9 | 0.292507 | 0.137375 | unchanged retained beam after two rounds |
| natural_84_1:18 | 0.147277 | 0.108797 | shared work budget during third round |

The sequential controls stopped after three rounds when their hypotheses stopped
changing. Both diagonal controls reached the work budget during their second
round. The diagonal decomposition costs extra reverse passes and admits less
search under the same cap; its total is not an equal-iteration performance claim.
The identical first-round states provide the narrower mechanism comparison.

## Frozen implementation and numerical qualification

Native code commit: `2406a11bfa83c7557622613cb4d3b15c9afa26ce`.
Base: `00c79da2856f0c787bf28a96ebc47be63083549b`.
Public resource: `meta-llama/Llama-3.2-1B-Instruct`, revision
`9213176726f574b556790deb65791e0c5aa438b6`, embedding plus layers0--3.
Exact prefix SHA256:
`10fd7e98719037954a51299eb9cb680544a659400ee2431e8aac333626f5e61c`.

The new engine connects provisional K/V through three singleton token calls while
keeping the committed cache detached and unchanged. This retains the corrected
BF16 native kernels, actual raw token input embeddings, positions, and FP32 RoPE.
There is no candidate batching, model fitting, continuous future embedding search,
learned initializer, or new committed token. Each retained hypothesis is a real
three-token block; beam<=2, two ranked proposals per slot, at most four rounds.
One reverse pass differentiates the sum of normalized observed residuals for
joint proposals. The local block control differentiates each slot's own loss.

The native port retains the prototype's fixed radius0.75 times median embedding
norm (0.6999441981), and uses the existing seed4401 random-token initializer plus
native full-vocabulary matvec/top32/direct-distance refinement. Incumbents are not
excluded from top2. This is a declared change from the old adaptive-radius8x8
solver, not an exact replay or a test of every possible native joint design. The
fixed radius and proposal width were not changed after seeing these failures.
A partial round returns the best actually checked block; unverified proposals
cannot become the returned answer.

The largest used context (18 committed tokens plus3 provisional) qualified before
scoring. Window outputs exactly matched sequential native commits, the root's own
gradient exactly matched the current-token path, and the root loss had zero
future-input gradients. Joint gradients included a nonzero later-observation
contribution. BF16 summed versus separately accumulated gradient relativeL2 was
0.0084694, below the predeclared0.02 engineering check; this is not bitwise gradient
equivalence or a token acceptance tolerance. Cache contents and parameter versions
stayed unchanged. Strict native allclose(atol=rtol=1e-5) was retained; the toy
FP64 threshold was not imported. Eight supplied synthetic tests passed in a copy.

## Work, timing, and resource boundaries

Five post-warmup measurements at one common BOS context priced native operations.
Median costs were2.518ms singleton forward,5.852ms window forward,8.087ms local
forward/backward,30.652ms joint window forward/backward,63.105ms diagonal
decomposition, and about2.2ms per vocabulary scan. Before either root was scored,
the work allowance was fixed to initial verification plus four rounds of local
gradient, one scan16 and16 singleton verifications: **0.204660 seconds per arm**.
The shared search wall cap was **0.419320 seconds per arm**. Every operation was
admitted against remaining work; actual wall was also checked. All six cells met
both caps. Calibration medians are an operation-cost ledger, not FLOPs or a claim
that all timing variation is explained. Forward rates exclude scalar verification
bookkeeping; actual search wall includes it. Both measures are preserved.

Over both roots, joint processed66 forward token positions,21 reverse-position
units and2,693,376 vocabulary rows; wider sequential processed75 forward positions,
6 reverse-position units and769,536 vocabulary rows. Diagonal reverse-position
counts are conservative whole-graph counts. None of these counts treats a window
as one ordinary token call.

Search timings include current-cell CUDA synchronization, proposals and actual
verification. They are single observations with counterbalanced arm order across
the two roots, not stable throughput estimates. The first joint cell includes
startup effects. The zero useful-candidate result does not depend on those timings.

The prediction job separately records0.488619s model loading,0.000402s observation
loading and0.083388s vocabulary preparation. Common prefix reconstruction took
0.332394s and0.033382s for the two contexts, including first-context startup. RNG,
observation transfer and warmup timings are retained per cell. The shared fixed
setup is charged identically to each arm in `total_charged_seconds`; that subtotal
excludes diagnostic warmups, hashing/imports and historical prefix search, so it
is not an end-to-end reconstruction time. Warmups took0.006521s and0.006294s.
The complete guarded calibration and prediction jobs took8.208s and6.137s,
including startup and watchdog sampling; post-import bodies took2.744s and2.395s.
No setup or previous reconstructed-history work is claimed to be free in a future
full reconstruction system.

Runtime: Python3.12.3, PyTorch2.10.0+cu128, Transformers5.3.0, RTX5080. Both jobs
used the independently restored public model and previously restored dependency
packages under Python-S. Fourteen actual backup/restore and dependency archive
files were rehashed; the exact execution source has a retained tar backup.
Peak GPU reserved memory was3,120,562,176 bytes (2.906GiB), below6GiB; peak measured
child host RSS stayed below2GiB. Both600s watchdogs passed, with no allocator,
thermal, memory or timeout failure. GPU processes were absent at completion.

## Artifacts and scope

The incoming native brief is preserved verbatim at
`coordination/requests/agent4-joint-native-gate.md`, with all25 supplied manifest
entries hash-verified under `experiments/agent4-joint-native-gate/incoming`.
The fixed plan, calibration, six complete candidate traces, freeze, separate
score, watchdogs and checks are under that experiment's evidence directory.
`manifest.json` binds their hashes and the code, model, observation and source IDs.

Replay in a fresh sibling worktree at the native code commit (whose calibration
and prediction outputs do not yet exist), using the exact commands in the two
guard receipts. Copy the final `scripts/agent4_joint/score.py` into that worktree
for scoring. Its sibling rescue worktree supplies the preserved assets. Outputs
are create-only. The final generic scorer was separately replayed against the
frozen evidence and reproduced every field except its scoring timestamp.
The scorer derives the gate decision from frozen lists; no native code changed
after execution. Existing datasets and independent restore assets remain external
local dependencies referenced by the manifest. The small result bundle is not a
model/dependency redistribution.

No full natural benchmark, new target training, recovered-prefix framework,
P03 holdout access, paid compute, registry activation, merge or publication occurred.
The old reconstruction results remain unchanged; canonical comparison is still
incomplete. This component supplies no rescue, overall speedup, target-tracking,
or equivalent-quality claim. No automatic follow-on is authorized.

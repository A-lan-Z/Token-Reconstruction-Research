# Bounded native joint-proposal component gate

Authorized by the user request to unzip and follow the supplied native brief.
Base: 00c79da2856f0c787bf28a96ebc47be63083549b. Create-only local study; no publication.

Use only the two opened missing roots: natural_11_1 position9 and natural_84_1
position18 (BOS is position0). Histories come from the frozen original discrete
predictions before each root, never from evaluator truth. Later observations are
permitted; future guessed IDs are the original seed4401 random initialization
stream, not future predictions or labels. No further positions are planned.

Common calibration context: natural_11_0 position1, BOS cache, three random real
tokens. No correctness scoring of calibration. Qualify the largest actual context
(position18, window3) before running either scored component. Reuse immutable
singleton native kernels and FP32 RoPE; graph-connected provisional K/V, committed
K/V detached and unchanged. No candidate batching or continuous future variables.

Three frozen arms: joint summed normalized residual gradient, own-observation
(diagonal) proposals with the same joint verification engine, and sequential
width16. Window3, beam<=2, two proposals/slot/beam, <=4 rounds. Use synthetic
radius0.75*median raw embedding norm for all arms, random native initializer,
and native matvec/top32/direct-distance refinement. This is a declared new
component configuration, not a replay of the older adaptive-radius 8x8 solver.
Incumbents retained, candidate tuples lexicographically ordered, best verified
block returned even after a partial budget stop. No token is committed by this
component diagnostic. Strict native allclose1e-5 is retained; no FP64 toy stop.

Measure five repetitions (after one warmup) of singleton/window forward,
joint/local/diagonal gradient and p2/p16 vocabulary scans at the common context.
Charge medians per real operation. Work cap is the measured cost of initial
singleton verification + four (local forward/backward + one scan16 +16 singleton
verifications). Shared wall cap is twice that work allowance plus0.01s for common
Python/control overhead. Every operation is admitted only if its work fits; wall
is checked before operations, and an overrun is reported and fails the gate.
No claim that median-cost accounting equals actual wall or FLOPs. Report both.

Model/dependency load, vocabulary preparation, per-context cache reconstruction,
RNG and observed-tensor preparation costs are recorded; identical shared setup is
charged to each arm for totals. Historical prefix search is supplied/sunk in this
opened-state component and was not rerun. It cannot be treated as reconstruction
time or an end-to-end speedup. Counterbalance mode order across the two roots.

Freeze all lists and traces before the separate scorer reads opened evaluator
labels. Report generated/verified/returned root inclusion, gained/lost versus
local block proposals and sequential, counters and measured costs. Stop if no
useful new root, if wider sequential matches/exceeds at no greater cost, or if
joint exceeds the shared budget. Favourable evidence permits only a later planned
study; no automatic expansion. Canonical comparisons remain incomplete.

Preflight: prior 127-position qualification used<4GiB reserved; planned largest
three connected provisional positions through four layers plus FP32 embedding
table and singleton candidates estimated<5GiB. Hard cap6GiB GPU reserved,
>=2GiB GPU free, >=8GiB host free, childRSS<=10GiB, temperature<80C; each GPU job
<=600s. Use existing independent restored public model and restored dependencies
under python -S. Hash actual backup bytes and archive exact source before runs.
Synthetic supplied tests are run only in a copy, with no cohort reruns needed.

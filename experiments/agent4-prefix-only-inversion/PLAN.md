# Prefix-only pilot: preregistered bounded development plan

Task: agent4-prefix-only-inversion. Base: 5bbc3bf42a81c814404cf84cb46d55f0d3418667.
Packet saved verbatim. No merges, global-registry edits, paid compute or P03 access.
This is task-local method registration; canonical dual matrix remains NOT RUN.

Read SipIt arXiv:2510.15511v4 and official source commit
820683156b7257313046a4fb3c492e52519525b7. The upstream guarantee allows up to
T*V candidate trials; this pilot explicitly does not inherit that guarantee.

Implement one sequential raw-embedding solver: random public token start,
continuous float32 embedding, MSE at current cut4 position, gradient norm cap1,
SGD lr1 with linear decay to .01 over 50 steps, nearest untried raw embedding
projection each step, snap every50. At most128 gradient steps and128 candidate
checks per token; best verified MSE token on exhaustion. No fitted proposals.
A small corrective variant requires diagnosed development failure. No sweep.
Full-prefix recomputation uses only own committed IDs, avoiding mutable trial
caches; backward, repeated prefix construction and vocabulary scans are timed.
BOS only known. Frozen prefix weights; no within-record model updates.

First run tiny synthetic CPU end-to-end, then public model diagnostics and
2 ordinary + 1 stress short development records (up to16 positions). Verify
raw inputs_embeds equality to discrete path, meaningful gradients, causal
positions/mask/cache, genuine-match residual for float32 and bf16 observations.
Observation generation and reconstruction run in separate processes. Public
source truth is evaluator-only until create-only prediction+trace freeze.
This development panel is never fresh confirmation. Freeze final configuration
before selecting unused public sources. Small final panel: 4 ordinary + 2 stress
records, initially16 positions; expand to128 only if development is useful.
Compare A1 top256 plus public-prefix direct cosine on same observations, own
committed prefix; label geometry/execution port and preserve decision constants.

Practical target: >=95% post-BOS accuracy, >=50% complete records, <=2x comparator
warmed reconstruction wall, peak GPU allocated<=6GiB; report all dimensions
separately and retain every failed/exhausted token in denominator.

Resource preflight: Llama vocab128256 x hidden2048, 4 decoder blocks. BF16 raw
embedding~0.49GiB, float32~0.98GiB; prefix float32~1.8GiB; scoring table~0.98GiB;
128x2048 hidden~1MiB, four-block backward activations conservatively<1GiB at
batch1; estimated GPU peak<5GiB (6GiB hard cap), CPU<10GiB. Require >=8GiB
GPU free before load, >=8GiB host available, >=20GiB disk, temperature<80C.
Qualify largest intended sequence before final panel; no candidate batching in
primary solver. No matrix until qualified. Fixed per-token time cap30s,
record cap600s; timed-out suffix abstains (-1), retained as wrong. Resource
anomalies abort fail-closed, not ordinary solver failures. Shared GPU lock to
be agreed with Agent2/3 before GPU work. Save actual prefix state, tokenizer,
lens comparator and source dependency archive, verify independent restore.

Stages2/3 require qualified target/recovered assets from Agent2. Locate and
verify metadata; no target adapter copying, no invented recovery framework.

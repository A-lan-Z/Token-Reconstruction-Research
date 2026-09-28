# Development33 causal decoder, frozen96-cell development grid

The eager72-cell and exact-replay72-cell public studies passed. Replay preserved
every original tensor for216runs and72eager buffer controls. All128native contexts
qualified; peak reserved5.863GiB. Current-token initialization+updates+final
evaluation took approximately10.18/16.27/28.54/53.16ms for1/2/4/8updates,
plus about1.4ms to commit. These suggest a remaining speed gap; do not claim a
fast replacement. The next question is whether causal history commitment repairs
the reconstruction accuracy enough to justify arithmetic optimization.

Freeze12exploratory methods: factors2/1/.5 x fixed updates8/4/2/1. Reconstruct the
unchanged eight retrospective development records,96cells, with only BOS known.
Each unknown position initializes all128256logits from the current H and public
prefix metric, optimizes the current soft embedding with the qualified FP32 map,
emits full-logit argmax, and commits its actual token embedding. Earlier emitted
IDs never change. No shortlist, token-proposal stage, separate candidate verifier,
offline fitted predictor, or target-prefix call. Skip the last emitted-token commit
because no later position consumes it. The method remains prefix-table dependent.

The exact qualified replay engine is inherited unchanged. Each factor runs in an
isolated restart-safe process and reuses its captures across four fixed update counts.
Capture all128native positions before evaluation. All input/output transfer,
per-position initialization, updates/evaluations, BOS/emitted-history commits,
diagnostics and synchronization enter inference time. Record prefix/lookup/capture
preparation separately and disclose sharing across update counts.

Before each factor's real outputs, qualify both lengths128/40 for all four counts:
three repeated whole-sequence decodes, one eager decode, exact all-output equality,
and a complete final K/V comparison with independent sequential forward commits from
the decoder's emitted IDs. These references use no source labels. Hash all per-layer
actual/reference histories and record the checks. Keep numerical/source/resource
guards and preserve failed attempts. No code/geometry changes after freeze.

Save emitted IDs, final per-position soft error/confidence, all eight-slot update
traces (unused slots zero), work counts, timing, memory, environment, commands and
asset/input/source hashes. The scorer requires all96cells and every public control,
then rejects missing-cell, changed-source and altered-output test cases before
opening labels. Score each condition/group separately; do not pool benchmarks.
Register an active method and run both canonical setups only if a suitable fixed
rule is selected after this exploratory study. The broader goal remains unmet.

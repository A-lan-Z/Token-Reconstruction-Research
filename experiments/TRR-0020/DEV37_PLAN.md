# Development37: full-vocabulary response translation, public diagnostic

The user requires no mechanism that narrows the vocabulary and proposes tokens
to A2. TRR-0019 does not satisfy that requirement. This study never passes
proposed token IDs to A2: it scores every128256cached response directly and emits
the final argmax. No fitted predictor or model-weight update is involved.

Prior distinction: TRR-0014 lookup_probe.py and dictionary_metric_probe.py use
the BOS response table to select16/64proposals, run native A2 and update the
query using the verified winner. This experiment instead predicts a single
context correction from fixed continuous probes that do not depend on the
current observed activation or its known public identity. No verified-winner
feedback, tokenizer fragments, oracle stopping or restricted token set is used.

Reuse the byte-bound BF16 native K256 BOS-response table from TRR-0014.
Its128256x2048FP32 cache occupies1,050,673,152bytes and originally took2.5557s
to generate, excluding IO. Hash-check the table and prefix. Reproduce both its
first and last256-row blocks exactly before using it. The preparation width256
is numerical batching over the entire vocabulary, not a shortlist. Do not
change its established build geometry. Rebuild is required on prefix changes.

Let S_i be that cached response. Define four response approximations:
1. raw: S_i;
2. mean: S_i + F_context(mean embedding) - F_BOS(mean embedding);
3. scaled_mean: same, after scaling the mean to RMS embedding norm;
4. four_probe: same, averaging the shifts from four fixed continuous probes.
The four probes use CPU seed37037, independent standard Gaussians scaled by the
per-coordinate embedding standard deviation, added to the mean, then rescaled
to RMS embedding norm. No probe is selected from token rankings or H.

Use both cosine and squared-distance final scores, always across the full
vocabulary. Qualify the factored cosine-norm and distance calculations against
explicit corrected vectors in independent tiny CPU cases, including cancellation.
Near-cancellation norms may be recomputed directly for numerical stability;
this is arithmetic on the same complete table, never A2 token proposals.

Use the same16explicitly public common-token fixtures as dev34, after four
fixed public histories: BOS alone and BOS plus deterministic random histories
of lengths15,63,127(seed37037+length). Target activations are generated with
native BF16 batch1current-token execution. This is a known-public numerical
test, not hidden-input reconstruction or a canonical benchmark.

The continuous-input adapter must exactly match native batch1on actual token
embeddings and an independent deep-copied-cache reference on continuous probes.
Preserve cache state. The cached K256responses may differ from batch1outputs;
quantify this, never claim output equivalence across those geometries.

Qualify the largest127-token context first. Freeze all four approximations and
two score rules on all64public observations, with three exact repeats, before
summarizing public identities. Preserve full scores, shifts, probes, target
vectors and dependency hashes. Charge context probe cost per unknown position;
do not amortize it across the16alternative public test observations when
describing prospective decoder cost. Record preparation and IO separately.
This test excludes emitted-history commits and is not end-to-end latency.

If public quality or cost is inadequate, do not promote a variant. If promising,
first implement/qualify a causal decoder, then register a fixed contender and
populate both canonical setups. No benchmark replacement claim is permitted
from this diagnostic. The supplied prefix is not a recovering-prefix trajectory.

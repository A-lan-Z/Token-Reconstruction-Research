# TRR-P06 implementation constraints from published parent

Status: implementation handoff after the complete packet was received. The
packet is preserved at `coordination/requests/TRR-P06.md` with SHA-256
`6f2883f1ec078877358c78fe5d05566ef845a1329f4e8f7a4a96aa69c5c5f992`. This
note records source and resource constraints; it does not authorize data
capture, fitting, truth opening, or GPU execution.

## Reusable parent interfaces

The reviewed parent at `f10f8ba438973b3cb260d41707fbb14293db9cd3` contains the
smallest suitable family in `src/token_reconstruction/trr0005_joint_decoder.py`:
`JointAffineAttentionDecoder` has trainable `W`, `b`, and log scale `s`, plus a
zero-output Q/K/V/output attention residual. Its public loader accepts BF16
`[records, 192, 2048]` observations, right-padded masks, public same-position
integer labels, and the normalized FP32 embedding table `[128256, 2048]`. The
same module already provides schedule, train-step, evaluation, and state I/O
helpers. P06 uses a separate module so that all three new visibility IDs have
one metadata-bound mask implementation.

The common production direct initialization is the pinned public affine state
from the corrected P06 plan:
`experiments/TRR-0004/evidence/affine/selected_states/fit_large_v1.historical_affine_ce_no_vocab_bias.safetensors`,
SHA-256
`09c5b852373d8555b06508a79bb00c94041202702b61b121b35fa2b6f9f64e65`.
It supplies the same public `W`, `b`, and `s` to every arm. The implementation
keeps an identity fallback only for model-free unit fixtures and labels it
`identity_fixture`; a production caller must pass the pinned state explicitly.

The published fitting script already fixes the useful recipe: shared public
schedule, 8-record batches, 512 post-BOS draws per update, 3,000 updates,
AdamW, learning rate `1e-3`, weight decay `0`, cosine schedule, gradient clip
`1`, validation every 100 steps, and earliest validation maximum including
step 0. Reuse these helpers rather than copying the full TRR-0005 prediction or
truth infrastructure. The prediction side has `run_warmed_prediction`,
`write_prediction_artifact`, and `write_prediction_receipt`; adapt its method
registration only after P06 IDs and panel are frozen.

## Minimal three-arm family

Use three new, explicit P06 method IDs with one implementation and one fixed
attention score mode shared by all arms:

- `positionwise`: diagonal visibility, only `H_i` is an allowed key for query
  `i` (the direct affine path remains active);
- `past_only`: causal visibility, `H_0 ... H_i` are allowed for query `i`;
- `full_record`: every valid activation `H_j` in the current record is allowed
  for every valid query `i`.

The full-record mask is the only new semantic branch: after constructing the
same `[B,T,T]` score tensor, use the valid keys in the current record with a
valid-query output mask. It must never admit right-padding, a different record,
or rows outside the declared H128 observation. P06 exposes a single explicit
visibility metadata field (`positionwise`, `past_only`, or `full_record`) and
rejects mismatched state metadata at load time. Keep the same Q/K/V/output
width and deterministic initialization stream in every arm; do not compare a
smaller full arm or a separately initialized model. Positionwise Q/K gradients
are expected to be inactive under a one-key diagonal mask and should be
reported as a capacity detail, while parameter counts, direct affine path,
optimizer, and update schedule stay identical.

At inference, each method receives the complete current-record activation
matrix before emitting any IDs. The full-record arm is explicitly a
full-record, non-streaming decoder. It must not read source IDs beyond BOS,
reconstructed-token feedback, correctness, guessed-prefix candidates, A2, or
any unavailable target computation. Prediction calls can still return the
whole sequence in left-to-right output order; later activation evidence may
influence an earlier output because it is an observation, not a label.

## Required focused checks before qualification

Use synthetic tensors with noncontiguous valid lengths and a padded suffix to
assert all of the following before any model load in a run:

1. Changing a later valid activation leaves earlier positionwise and past-only
   logits unchanged.
2. Changing a later valid activation can change an earlier full-record logit
   when the residual output is nonzero.
3. Changing padded values changes no valid output for all arms.
4. Records cannot attend across one another, and invalid query rows stay zero.
5. All three arms have identical parameter names/counts and the same direct
   affine initial function; the visibility mode is present in state metadata.
6. The training path uses the declared mask for every arm rather than loading a
   causal state and merely removing a mask during evaluation.
7. Full-record outputs are deterministic and repeated prediction IDs are exact
   matches; no candidate arrays or A2 calls are produced.

A public fitting diagnostic should run the pinned competent affine
initialization first, freeze a count-only set of public fitting positions it
gets wrong, and show that each added path improves on those rows after a fixed
probe. The error audit is from the permitted public fitting labels and must be
fixed before seeing validation or fresh-panel results; it must not select a
winning arm. Probe states are discarded before the six main fits.

## Data, evaluation, and resource constraints

Prefer one predeclared public fit distribution with varied coverage (the
published `enriched` bank is the natural candidate), or run every arm on both
banks if design explicitly requires it. In either case use identical records,
labels, masks, schedule, optimizer, checkpoints, and selection opportunities
within each arm. Do not use any P04 teacher-ranking loss, P03 projected
prototypes, or sealed holdout.

The fresh panel must be selected separately from fitting/validation, with the
same observations and source order for all three methods and a paired changed
target when setup can provide it. Freeze states, code, panel, observations,
method registration, and every prediction before opening truth. Report each
domain and target separately: unrestricted token accuracy, exact clip rate,
paired token gains/regressions, exact-record events, position bins including
record ends, and source-record uncertainty. Keep the A1+A2 anchor as a separate
bounded denominator; students remain activation-only and search-free.

The source artifacts remain `[records,192,2048]`, but corrected P06 fit,
validation, capacity-probe, and evaluation code crops every split to H128
before schedule construction and metrics. The largest backward geometry is
therefore record batch 8, sequence 128, hidden 2048, full vocabulary 128256,
selected position budget 512, context width 128, FP32 embedding/logit math
with BF16 observations. The full visibility mask has the same dense `B*T*T`
score workspace as the parent causal implementation. Use a source-only
geometry estimate, then qualify one 8x128 backward cell before the matrix.
Retain the P05 live limits of at least 8 GiB free GPU, at most 6 GiB PyTorch
reserved GPU, at most 16 GiB whole-process RSS, and at least 10 GiB host
available memory, plus the external fail-closed watchdog. Do not use a
smaller projection chunk or microbatch as a workaround without a bitwise or
declared output-equivalence check.

The published public table is 1,050,673,488 bytes (`[128256,2048]` FP32), and
P05's matching largest backward run measured below the 6 GiB reserved-device
limit. P06 must record its own child and watchdog peaks, phases, exact command,
source commit, assets, and timing; prior measurements are an estimate only.

## Runner interface (source only; no P06 fit has run)

The authoritative executable is `scripts/trr0006_fit_visibility.py`. It has
three create-only modes. `preflight` reads manifest metadata plus the pinned
affine state and writes `preflight.json`; `probe` requires that passing receipt,
loads only the public fit/validation bank, builds the 256-error ledger and
fixed 300-update probe, and writes `capacity_probe_receipt.json`; `main`
requires both receipts and runs the six 3,000-update fits. Probe/main reject
any direct-affine hash other than the registered value and reject a preflight
whose manifest or H128 geometry binding differs.

The source-only command shape is:

```text
PYTHONPATH=src:scripts python3 scripts/trr0006_fit_visibility.py --mode preflight --fit-manifest experiments/TRR-0005/public_activation_v1/enriched_manifest.json --direct-affine-state experiments/TRR-0004/evidence/affine/selected_states/fit_large_v1.historical_affine_ce_no_vocab_bias.safetensors --output-root <fresh-preflight-root> --device cpu
```

Probe and main use the same arguments plus `--preflight-receipt`, with fresh
output roots and `--device cuda` only after the declared resource window and
largest-cell qualification are approved. Main additionally supplies the PASS
`--probe-receipt`. Both execution modes default to four Torch intra-op and one
inter-op thread, 8 GiB minimum free GPU, 6 GiB maximum reserved GPU, 16 GiB
process RSS, 10 GiB host available memory, and an 1,800-second deadline.
